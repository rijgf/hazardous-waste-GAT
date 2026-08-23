param(
    [string]$OutputDir = (Split-Path -Parent $PSCommandPath),
    [string]$OutputName = 'PPO改进流程图_可编辑重绘.vsdx'
)

$ErrorActionPreference = 'Stop'

$script:PageW = 17.5
$script:PageH = 10.0
$script:RefW = 1792.0
$script:RefH = 1024.0

function RGBF([int]$r, [int]$g, [int]$b) { "RGB($r,$g,$b)" }
function VX([double]$x) { $script:PageW * $x / $script:RefW }
function VY([double]$y) { $script:PageH - ($script:PageH * $y / $script:RefH) }

$C = @{
    Blue = RGBF 17 84 184
    Blue2 = RGBF 29 101 205
    PaleBlue = RGBF 239 247 255
    DeepBlue = RGBF 24 91 188
    Orange = RGBF 211 92 26
    PaleOrange = RGBF 255 242 230
    Gray = RGBF 90 90 90
    Black = RGBF 16 16 16
    White = RGBF 255 255 255
}

function Set-Cell($shape, [string]$cell, [string]$formula) {
    try { $shape.CellsU($cell).FormulaU = $formula } catch {}
}

function Style-Shape($shape, [string]$fill, [string]$line, [double]$linePt = 1.0, [int]$dash = 1, [double]$round = 0.06) {
    if ($fill -eq 'none') {
        Set-Cell $shape 'FillPattern' '0'
    } else {
        Set-Cell $shape 'FillPattern' '1'
        Set-Cell $shape 'FillForegnd' $fill
    }
    if ($line -eq 'none') {
        Set-Cell $shape 'LinePattern' '0'
    } else {
        Set-Cell $shape 'LinePattern' ([string]$dash)
        Set-Cell $shape 'LineColor' $line
        Set-Cell $shape 'LineWeight' "$linePt pt"
    }
    if ($round -gt 0) { Set-Cell $shape 'Rounding' "$round in" }
}

function Set-Text($shape, [string]$text, [double]$size = 12, [string]$color = $C.Black, [bool]$bold = $false, [int]$align = 1) {
    $shape.Text = $text
    Set-Cell $shape 'Char.Font' 'FONT("Microsoft YaHei UI")'
    Set-Cell $shape 'Char.Size' "$size pt"
    Set-Cell $shape 'Char.Color' $color
    Set-Cell $shape 'Char.Style' ($(if ($bold) { '1' } else { '0' }))
    Set-Cell $shape 'Para.HorzAlign' ([string]$align)
    Set-Cell $shape 'VerticalAlign' '1'
    foreach ($m in 'TxtMarginLeft','TxtMarginRight','TxtMarginTop','TxtMarginBottom') {
        Set-Cell $shape $m '2 pt'
    }
}

function RectTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text = '', [string]$fill = $C.PaleBlue, [string]$line = $C.Blue, [double]$size = 12, [bool]$bold = $false, [double]$linePt = 1.2, [int]$dash = 1, [double]$round = 0.08) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s $fill $line $linePt $dash $round
    if ($text -ne '') { Set-Text $s $text $size $C.Black $bold }
    return $s
}

function TextTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$size = 12, [string]$color = $C.Black, [bool]$bold = $false, [int]$align = 1) {
    $s = RectTL $x $y $w $h '' 'none' 'none' $size $bold 0 1 0
    Set-Text $s $text $size $color $bold $align
    return $s
}

function LineTL([double]$x1, [double]$y1, [double]$x2, [double]$y2, [string]$color = $C.Black, [double]$linePt = 1.0, [bool]$arrowEnd = $false, [int]$dash = 1) {
    $s = $script:Page.DrawLine((VX $x1), (VY $y1), (VX $x2), (VY $y2))
    Set-Cell $s 'LineColor' $color
    Set-Cell $s 'LineWeight' "$linePt pt"
    Set-Cell $s 'LinePattern' ([string]$dash)
    if ($arrowEnd) { Set-Cell $s 'EndArrow' '4' }
    return $s
}

function DiamondTL([double]$cx, [double]$cy, [double]$w, [double]$h, [string]$text) {
    $s = $script:Page.DrawOval((VX ($cx - 1)), (VY ($cy + 1)), (VX ($cx + 1)), (VY ($cy - 1)))
    $s.Delete()
    $shape = $script:Page.DrawRectangle((VX ($cx - $w / 2)), (VY ($cy + $h / 2)), (VX ($cx + $w / 2)), (VY ($cy - $h / 2)))
    $shape.CellsU('Angle').FormulaU = '45 deg'
    Style-Shape $shape $C.PaleBlue $C.Blue 1.2 1 0
    if ($text -ne '') { Set-Text $shape $text 12 $C.Black $true }
    return $shape
}

function FlowBox([double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$size = 13) {
    RectTL $x $y $w $h $text $C.PaleBlue $C.Blue $size $true 1.25 1 0.08 | Out-Null
}

function Note([double]$x, [double]$y, [double]$w, [string]$text) {
    TextTL $x $y $w 75 $text 9.5 $C.Black $false | Out-Null
}

function ArrowBetween([double]$x1, [double]$y, [double]$x2) {
    LineTL $x1 $y $x2 $y $C.Black 1.1 $true | Out-Null
}

function StageLabel([double]$x, [double]$y, [double]$w, [string]$text) {
    TextTL $x ($y - 42) $w 32 $text 12.5 $C.Blue $true | Out-Null
    LineTL $x $y ($x + $w) $y $C.Blue 1.0 $false 2 | Out-Null
    RectTL ($x - 5) ($y - 5) 10 10 '' $C.Blue $C.Blue 4 $false 0.4 1 0.05 | Out-Null
    RectTL ($x + $w - 5) ($y - 5) 10 10 '' $C.Blue $C.Blue 4 $false 0.4 1 0.05 | Out-Null
}

function BracketLabel([double]$x, [double]$y, [double]$w, [string]$text) {
    TextTL $x ($y - 58) $w 35 $text 12.5 $C.Blue $true | Out-Null
    LineTL $x $y ($x + $w) $y $C.Blue 1.0 $false | Out-Null
    LineTL $x $y ($x + 10) ($y + 18) $C.Blue 1.0 $false | Out-Null
    LineTL ($x + $w) $y ($x + $w - 10) ($y + 18) $C.Blue 1.0 $false | Out-Null
    LineTL ($x + $w / 2 - 12) $y ($x + $w / 2) ($y - 18) $C.Blue 1.0 $false | Out-Null
    LineTL ($x + $w / 2 + 12) $y ($x + $w / 2) ($y - 18) $C.Blue 1.0 $false | Out-Null
}

function Draw-TrainingPage {
    TextTL 385 35 1020 85 '训练阶段：学习“如何改进”' 27 $C.Black $true | Out-Null
    TextTL 530 132 740 45 '通过奖励反馈训练出高质量局部搜索策略' 15 $C.Gray $false | Out-Null

    StageLabel 52 324 290 '构造训练环境'
    StageLabel 470 324 355 '识别当前解结构'
    StageLabel 912 324 160 '选择改进行动'
    StageLabel 1128 324 240 '生成并评估候选解'
    StageLabel 1436 324 176 '奖励驱动学习'

    $y = 388; $h = 205
    FlowBox 20 $y 145 $h "训练问题`n实例"
    FlowBox 220 $y 132 $h "贪心`n初始解"
    FlowBox 410 $y 120 $h "当前解`n状态"
    RectTL 570 354 255 282 '' $C.DeepBlue $C.Blue 15 $true 1.2 1 0.08 | Out-Null
    TextTL 600 382 195 36 'PPO改进智能体' 14 $C.White $true | Out-Null
    RectTL 590 438 220 44 'State Encoder' $C.White $C.White 10.5 $true 0.6 1 0.04 | Out-Null
    RectTL 590 505 220 44 'Transformer' $C.White $C.White 10.5 $true 0.6 1 0.04 | Out-Null
    RectTL 590 572 220 44 'Policy / Value Heads' $C.White $C.White 10.5 $true 0.6 1 0.04 | Out-Null
    LineTL 700 482 700 505 $C.White 1.0 $true | Out-Null
    LineTL 700 549 700 572 $C.White 1.0 $true | Out-Null

    FlowBox 870 $y 130 $h "选择局部`n搜索算子`n与对象" 12.5
    FlowBox 1050 $y 132 $h "执行`n局部搜索"
    FlowBox 1225 $y 110 $h "修复与`n评估"
    FlowBox 1370 $y 95 $h "计算`n奖励"
    FlowBox 1505 $y 105 $h "PPO`n参数更新" 12
    RectTL 1648 $y 128 $h "训练完成的`nPPO改进`n策略" $C.PaleOrange $C.Orange 12.5 $true 1.25 1 0.08 | Out-Null

    ArrowBetween 165 490 220
    ArrowBetween 352 490 410
    ArrowBetween 530 490 570
    ArrowBetween 825 490 870
    ArrowBetween 1000 490 1050
    ArrowBetween 1182 490 1225
    ArrowBetween 1335 490 1370
    ArrowBetween 1465 490 1505
    ArrowBetween 1610 490 1648

    Note 26 660 120 "输入问题数据，`n构建实例"
    Note 228 660 120 "使用贪心方法`n生成初始解"
    Note 412 660 140 "表示当前解的`n结构与特征"
    Note 604 660 190 "编码解结构，理解状态，`n输出策略与价值评估"
    Note 880 660 126 "基于策略选择`n改进行动"
    Note 1068 660 120 "应用算子生成`n候选解"
    Note 1235 660 110 "修复候选解并`n评估其质量"
    Note 1372 660 105 "根据改进效果`n计算奖励信号"
    Note 1502 660 120 "利用奖励信号`n更新智能体参数"
    Note 1650 660 115 "获得高质量的`n改进策略"

    LineTL 468 644 468 588 $C.Blue 1.2 $true | Out-Null
    LineTL 1558 594 1558 845 $C.Blue 1.2 $false | Out-Null
    LineTL 468 845 468 714 $C.Blue 1.2 $false | Out-Null
    LineTL 468 845 1558 845 $C.Blue 1.2 $false | Out-Null
    RectTL 800 814 180 45 '反复交互学习' $C.White $C.Blue 12.5 $true 1.2 1 0.04 | Out-Null
}

function Draw-SolvingPage {
    TextTL 382 35 1028 82 '求解阶段：利用所学策略指导启发式改进' 27 $C.Black $true | Out-Null
    TextTL 500 127 790 45 '面对新实例时固定策略参数，仅利用训练好的智能体进行迭代优化' 15 $C.Gray $false | Out-Null

    TextTL 708 218 170 55 '来自训练阶段' 11.5 $C.Black $false | Out-Null
    RectTL 708 218 170 55 '' 'none' $C.Blue 4 $false 1.0 1 0.05 | Out-Null
    LineTL 793 273 793 405 $C.Blue 1.0 $true 2 | Out-Null

    BracketLabel 55 380 350 '构造初始可行解'
    BracketLabel 470 380 400 '调用已训练策略'
    BracketLabel 925 380 340 '执行启发式改进'
    BracketLabel 1315 380 425 '接受优质候选解'

    $y = 430; $h = 120
    FlowBox 34 $y 168 $h '新问题实例' 12.5
    FlowBox 252 $y 158 $h '贪心初始解' 12.5
    FlowBox 460 $y 146 $h '当前解编码' 12.5
    RectTL 645 420 235 142 '' $C.DeepBlue $C.Blue 12.5 $true 1.2 1 0.08 | Out-Null
    TextTL 650 438 225 48 '训练好的PPO改进策略' 12.5 $C.White $true | Out-Null
    RectTL 665 500 190 42 '参数固定，不更新' $C.White $C.White 10.5 $true 0.6 1 0.04 | Out-Null
    FlowBox 922 $y 170 $h "选择局部`n搜索算子与对象" 11.5
    FlowBox 1138 $y 155 $h '执行局部搜索' 12.5
    FlowBox 1342 $y 165 $h '修复与评估' 12.5
    FlowBox 1558 $y 175 $h "更新当前解`n与最优解" 12.0

    ArrowBetween 202 490 252
    ArrowBetween 410 490 460
    ArrowBetween 606 490 645
    ArrowBetween 880 490 922
    ArrowBetween 1092 490 1138
    ArrowBetween 1293 490 1342
    ArrowBetween 1507 490 1558

    LineTL 1645 550 1645 617 $C.Black 1.1 $true | Out-Null
    DiamondTL 1645 660 172 112 '' | Out-Null
    TextTL 1588 626 114 68 "是否达到`n停止条件？" 12 $C.Black $true | Out-Null
    LineTL 1645 716 1645 830 $C.Black 1.1 $true | Out-Null
    TextTL 1675 760 40 32 '是' 12 $C.Black $true | Out-Null
    RectTL 1490 832 295 86 '输出最终优化方案' $C.PaleOrange $C.Orange 14 $true 1.2 1 0.08 | Out-Null

    LineTL 1559 660 530 660 $C.Black 1.0 $false | Out-Null
    LineTL 530 660 530 550 $C.Black 1.0 $true | Out-Null
    TextTL 1345 625 44 30 '否' 12 $C.Black $true | Out-Null
    TextTL 870 678 215 36 '继续迭代改进' 12.5 $C.Blue $true | Out-Null
}

$vsdxPath = Join-Path $OutputDir $OutputName
$exportDir = Join-Path $OutputDir 'exports'
New-Item -ItemType Directory -Force -Path $exportDir | Out-Null

$backup = $null
if (Test-Path -LiteralPath $vsdxPath) {
    $backup = Join-Path $OutputDir (([IO.Path]::GetFileNameWithoutExtension($vsdxPath)) + '.backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.vsdx')
    Copy-Item -LiteralPath $vsdxPath -Destination $backup
}

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $doc = $visio.Documents.Add('')

    $script:Page = $doc.Pages.Item(1)
    $script:Page.Name = '训练阶段'
    $script:Page.PageSheet.CellsU('PageWidth').FormulaU = "$script:PageW in"
    $script:Page.PageSheet.CellsU('PageHeight').FormulaU = "$script:PageH in"
    Draw-TrainingPage

    $script:Page = $doc.Pages.Add()
    $script:Page.Name = '求解阶段'
    $script:Page.PageSheet.CellsU('PageWidth').FormulaU = "$script:PageW in"
    $script:Page.PageSheet.CellsU('PageHeight').FormulaU = "$script:PageH in"
    Draw-SolvingPage

    $doc.SaveAs($vsdxPath) | Out-Null

    $doc.Pages.Item('训练阶段').Export((Join-Path $exportDir '训练阶段.png')) | Out-Null
    $doc.Pages.Item('求解阶段').Export((Join-Path $exportDir '求解阶段.png')) | Out-Null
    $doc.Pages.Item('训练阶段').Export((Join-Path $exportDir '训练阶段.svg')) | Out-Null
    $doc.Pages.Item('求解阶段').Export((Join-Path $exportDir '求解阶段.svg')) | Out-Null

    Write-Output "VSDX=$vsdxPath"
    if ($backup) { Write-Output "BACKUP=$backup" } else { Write-Output 'BACKUP=(new file; no prior target existed)' }
    Write-Output "PNG1=$(Join-Path $exportDir '训练阶段.png')"
    Write-Output "PNG2=$(Join-Path $exportDir '求解阶段.png')"
    Write-Output "SVG1=$(Join-Path $exportDir '训练阶段.svg')"
    Write-Output "SVG2=$(Join-Path $exportDir '求解阶段.svg')"
    Write-Output "PAGES=$($doc.Pages.Count)"
    Write-Output "SHAPES_TRAINING=$($doc.Pages.Item('训练阶段').Shapes.Count)"
    Write-Output "SHAPES_SOLVING=$($doc.Pages.Item('求解阶段').Shapes.Count)"
} finally {
    if ($doc -ne $null) {
        try { $doc.Close() } catch {}
    }
    if ($visio -ne $null) {
        try { $visio.Quit() } catch {}
    }
}
