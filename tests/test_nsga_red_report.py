"""Template-only fixture test; never publishes or fabricates experiment records."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_nsga_red_budget as report


class ReportTemplateTests(unittest.TestCase):
    def test_template_in_temporary_directory(self):
        # Use existing historical archives solely as a labelled test fixture.
        original_front=report.front
        original_read=report.read
        def fixture_front(root,name):
            return original_front(report.BASE,name)
        def fixture_read(path):
            if Path(path).parent.name=='completed': return {'files':{}}
            return original_read(path)
        with patch.object(report,'front',fixture_front),patch.object(report,'read',fixture_read):
            fixture=report.data()
        before=(report.OUT/'before'/report.PAPER.name).read_text(encoding='utf8')
        with tempfile.TemporaryDirectory(prefix='nsga-report-test-') as tmp:
            out=Path(tmp);(out/'before').mkdir()
            (out/'before'/report.PAPER.name).write_text(before,encoding='utf8')
            with patch.object(report,'data',return_value=fixture),patch.object(report,'OUT',out),contextlib.redirect_stdout(io.StringIO()):
                report.build()
            text=(out/report.PAPER.name).read_text(encoding='utf8')
            self.assertIn('种群20',text)
            self.assertNotIn('NSGA-II（B1；v4复用）',text)
            self.assertIn('![PPO与NSGA-II前沿对比]',text)
            self.assertTrue((out/'figures/ppo_nsga_best.png').exists())
