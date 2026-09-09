"""Keep editable LaTeX equations and portable GFM tables in the manuscript."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '供应链管理写作/数值实验与结果分析_论文稿.md'


class PaperMarkdownCompatibilityTests(unittest.TestCase):
    def test_published_paper_uses_editable_latex_and_gfm_tables(self):
        paper = PAPER.read_text(encoding='utf8')
        with self.subTest(feature='no raw HTML tables'):
            self.assertIsNone(re.search(r'<\s*/?\s*(?:table|thead|tbody|tr|th|td)\b', paper, re.I))
        with self.subTest(feature='three editable display equations'):
            blocks = re.findall(r'^\$\$\n(.*?)\n\$\$$', paper, re.M | re.S)
            self.assertEqual(len(blocks), 3)
        for formula in ('目标函数', '弱覆盖率', '相对差异'):
            with self.subTest(formula=formula):
                self.assertNotIn(f'![{formula}]', paper)
                tex = (ROOT / '供应链管理写作/排版资源' / f'{formula}.tex').read_text(encoding='utf8')
                equation = tex.split(r'\[')[1].split(r'\]')[0].strip()
                self.assertIn(equation, blocks)
        expected = {
            '表5': (1, ['B1 N→P', 'B1 P→N', 'B2 N→P', 'B2 P→N']),
            '表6': (2, [f'Test-{scale} {direction}' for scale in range(1, 5)
                       for direction in ('N→P', 'P→N')]),
        }
        for name, (data_rows, headers) in expected.items():
            with self.subTest(table=name):
                section = re.search(r'\*\*' + name + r'[^\n]*\n\n([^\n]*(?:\n[^\n]+)*)', paper)
                self.assertIsNotNone(section)
                lines = section.group(1).splitlines()
                self.assertEqual(len(lines), data_rows + 2)
                cells = [[cell.strip() for cell in line.strip().strip('|').split('|')]
                         for line in lines]
                self.assertTrue(all(len(row) == len(headers) + 1 for row in cells))
                self.assertEqual(cells[0][1:], headers)
        with self.subTest(feature='hashes do not stretch table cells'):
            table_lines = '\n'.join(line for line in paper.splitlines() if line.startswith('|'))
            self.assertIsNone(re.search(r'\b[a-f0-9]{64}\b', table_lines))


if __name__ == '__main__':
    unittest.main()
