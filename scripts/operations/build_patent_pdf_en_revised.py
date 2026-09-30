"""Build the revised working specification from its Markdown source.

Uses ReportLab; preserves the original submitted-for-review PDF and its builder.
Run with the bundled Python runtime. Taxonomy tables are expanded once from
the repository snapshot; normal builds do not rewrite the specification.
"""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'patents'
SOURCE = OUT / 'patent_specification_engine_bay_inspection_en_revised.md'
PDF = OUT / 'patent_specification_engine_bay_inspection_en_revised.pdf'
HTML = OUT / 'patent_specification_en_revised.html'
TITLE_SECTION = '## 1. TITLE OF THE INVENTION'
ABSTRACT_SECTION = '## 8. ABSTRACT'
ABSTRACT_MAX_WORDS = 150
HTML_LANG = 'en'


def taxonomy_tables():
    text = (ROOT / 'configs' / 'taxonomy_v2.yaml').read_text(encoding='utf-8')
    systems = []
    section = text.split('systems:\n', 1)[1].split('\ngeneric_classes:', 1)[0]
    for line in section.splitlines():
        m = re.match(r'  (\w+):.*en: "([^"]+)"', line)
        if m:
            systems.append(m.groups())
    generic = []
    section = text.split('generic_classes:\n', 1)[1].split('\ndetail_classes:', 1)[0]
    for line in section.splitlines():
        m = re.match(r'  (\w+):.*en: "([^"]+)"', line)
        if m:
            generic.append(m.groups())
    components = []
    section = text.split('components:\n', 1)[1]
    for match in re.finditer(r'^  (\w+):\s*\n(.*?)(?=^  \w+:|\Z)', section, re.M | re.S):
        key, body = match.groups()
        fields = {}
        for name in ('system', 'tier', 'fallback'):
            m = re.search(r'^    ' + name + r':\s*([^\n#]+)', body, re.M)
            fields[name] = m.group(1).strip() if m else 'unspecified'
        components.append((key, fields['system'], fields['tier'], fields['fallback']))
    assert systems and generic and components
    def mdtable(title, headers, rows):
        return title + '\n\n| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n\n'.join([
        f'Taxonomy snapshot: {len(systems)} systems, {len(components)} fine-component records, and {len(generic)} fallback identifiers. Counts describe this version and do not limit the claims.',
        mdtable('Table 1. Functional system identifiers', ['Identifier', 'System'], systems),
        mdtable('Table 2. Fine identity and declared training-state mapping', ['Fine identity', 'System', 'State', 'Fallback'], components),
        mdtable('Table 3. Fallback identifiers', ['Identifier', 'Description'], generic),
    ])


def markup(text):
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    return text


styles = getSampleStyleSheet()
styles.add(ParagraphStyle('BodyPatent', fontName='Times-Roman', fontSize=10.5,
                          leading=14, alignment=TA_JUSTIFY, spaceAfter=7,
                          allowWidows=False, allowOrphans=False))
styles.add(ParagraphStyle('PatentTitle', fontName='Times-Bold', fontSize=14,
                          leading=18, alignment=TA_CENTER, spaceAfter=16))
styles.add(ParagraphStyle('PatentH2', fontName='Times-Bold', fontSize=12,
                          leading=15, spaceBefore=13, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle('PatentH3', fontName='Times-Bold', fontSize=11,
                          leading=14, spaceBefore=10, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle('CellPatent', fontName='Times-Roman', fontSize=8.5,
                          leading=11, wordWrap='CJK'))
styles.add(ParagraphStyle('CaptionPatent', fontName='Times-Bold', fontSize=10,
                          leading=13, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle('FigurePatent', fontName='Times-Roman', fontSize=11,
                          leading=15, alignment=TA_CENTER))
styles.add(ParagraphStyle('CodePatent', fontName='Courier', fontSize=8,
                          leading=10, spaceAfter=10))

WIDTH = A4[0] - 45 * mm


def table(rows):
    count = len(rows[0])
    # Component identifiers need space and may wrap at underscores.
    weights = [0.31, 0.25, 0.08, 0.36] if count == 4 and rows[0][0] == 'Fine identity' else [1 / count] * count
    widths = [WIDTH * w for w in weights]
    cells = [[Paragraph(markup(c).replace('_', '_<wbr/>'), styles['CellPatent']) for c in row] for row in rows]
    obj = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return obj


FIGURES = [
    ('FIG. 1. Inspection architecture (100)', [
        'Image acquisition (102): image + optional vehicle metadata',
        'Multi-scale preprocessing (104): global image + overlapping regions',
        'Segmentation network (106): class, box, mask, branch provenance',
        'Common coordinates + duplicate processing (110)',
        'Spatial evidence and count constraints (108) qualify instances',
        'Expectation reasoning (112): observed / expected but not observed',
        'Display and structured output (114): uncertainty + review status']),
    ('FIG. 2. Hierarchy and training-state axis (200)', [
        'Hierarchy: functional system -> morphological group -> fine identity',
        'B: candidate fine identity; training maps to C fallback',
        'B -> A: reviewed real count >=60 AND distinct vehicles >=8',
        'A: independent training class in a versioned model manifest',
        'A -> B / C mapping: reviewed real count <40',
        'Otherwise retain state; synthetic instances excluded from counts']),
    ('FIG. 3. Multi-scale coordinate and merge pipeline (300)', [
        'Global branch (302) + overlapping local branches (304)',
        'Record crop origin, resize scale, and letterbox padding',
        'Inverse resize + crop offset for BOTH box corners and mask points',
        'Internal-border truncation handling; common image coordinates (306)',
        'Class-wise NMS -> class-agnostic NMS -> supported count cap (308)',
        'Spatial evidence is soft for unregistered views; preserve audit reasons']),
    ('FIG. 4. Verdict-based curation schematic (400)', [
        'Features (402): VLM score; source/class; crop similarity margin',
        'Geometry, spatial density, size, borders, overlaps, candidate counts',
        'GBDT (404): class-right score; optional separate probability calibration',
        'Vehicle-separated reference banks, validation, and final evaluation',
        'Admission gates: class score + accepted geometry + uniqueness',
        'Admit with provenance / expert review / exclude from training',
        'Historical AUROC 0.844 vs 0.617; no reconstructed ROC curve (406)']),
    ('FIG. 5. Observations and vehicle expectations (500)', [
        'Visible component hierarchy (502): boxes, masks, identities, scores',
        'Resolve YMME/options against versioned configuration knowledge',
        'Expected not observed (504): uncertainty and unresolved generic links',
        'Non-observation does not establish absence or malfunction',
        'Estimated region only with supported diagram/landmark registration',
        'Separate supplied DTC + applicable inspection guidance (506)']),
    ('FIG. 6. Optional offline synthetic-data engine (600)', [
        'Assets (602): taxonomy-linked component meshes / configuration scenes',
        'Randomization (604): camera, lights, materials, grime, distractors',
        'Renderer (606): RGB + instance IDs; optional unoccluded reference',
        'Ground truth (608): visible masks, tight boxes, labels, visibility gate',
        'Mixer (610): low-support classes; synthetic provenance retained',
        'Training (612): pretraining -> real-image fine-tuning -> network (106)',
        'Offline rendering; synthetic counts excluded from real promotion']),
    ('FIG. 7. Illustrative projection and duplicate removal', [
        'Image: 1600 x 1200; grid 2 x 2; overlap 0.25',
        'Lower-right crop origin (685,514); local box (100,80,160,140)',
        'Global box (785,594,845,654); apply same offset to mask vertices',
        'Global candidate (786,595,846,655); box IoU approximately 0.936',
        'Scores 0.78 vs 0.71; NMS 0.85 retains the tile box and mask',
        'Class-right score 0.92; geometry review remains required',
        'ILLUSTRATIVE ONLY: values are not empirical measurements']),
]


def drawing_pages():
    story = [PageBreak(), Paragraph('DRAWINGS', styles['PatentH2'])]
    for index, (title, steps) in enumerate(FIGURES):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(title, styles['PatentH2']))
        for n, step in enumerate(steps):
            if n:
                story.append(Paragraph('&#8595;', styles['FigurePatent']))
            block = Table([[Paragraph(markup(step), styles['FigurePatent'])]], colWidths=[WIDTH - 20 * mm])
            block.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 0.8, colors.black),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f4f4f4')),
                ('TOPPADDING', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 11),
            ]))
            story.append(block)
        story.append(Spacer(1, 12))
        story.append(Paragraph('Schematic embodiment. See the corresponding detailed description; no performance curve is inferred from this drawing.', styles['BodyPatent']))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman', 8)
    canvas.drawString(25 * mm, 12 * mm, 'Revised working draft - 30 September 2026')
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def main():
    text = SOURCE.read_text(encoding='utf-8')
    if '{{TAXONOMY_TABLES}}' in text:
        text = text.replace('{{TAXONOMY_TABLES}}', taxonomy_tables())
        SOURCE.write_text(text, encoding='utf-8')
    lines = text.splitlines()
    story, web = [], []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            story.append(Preformatted('\n'.join(code), styles['CodePatent']))
            web.append('<pre>' + html.escape('\n'.join(code)) + '</pre>')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', c) for c in row):
                    rows.append(row)
                i += 1
            rendered_table = table(rows)
            story.extend([KeepTogether([rendered_table]) if len(rows) <= 10 else rendered_table, Spacer(1, 10)])
            web.append('<table>' + ''.join('<tr>' + ''.join('<td>' + markup(c) + '</td>' for c in r) + '</tr>' for r in rows) + '</table>')
            continue
        elif line.startswith('# '):
            story.append(Paragraph(markup(line[2:]), styles['PatentTitle']))
            web.append('<h1>' + markup(line[2:]) + '</h1>')
        elif line.startswith('## '):
            if line.startswith(('## 7.', '## 8.')):
                story.append(PageBreak())
            story.append(Paragraph(markup(line[3:]), styles['PatentH2']))
            web.append('<h2>' + markup(line[3:]) + '</h2>')
        elif line.startswith('### '):
            story.append(Paragraph(markup(line[4:]), styles['PatentH3']))
            web.append('<h3>' + markup(line[4:]) + '</h3>')
        else:
            style = styles['CaptionPatent'] if re.match(r'^Table \d+\.', line) else styles['BodyPatent']
            paragraph = Paragraph(markup(line), style)
            story.append(KeepTogether([paragraph]) if re.match(r'^\d+\. ', line) else paragraph)
            web.append('<p>' + markup(line) + '</p>')
        i += 1
    story.extend(drawing_pages())
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=25*mm,
                           rightMargin=20*mm, topMargin=18*mm, bottomMargin=20*mm,
                           title=text.split(TITLE_SECTION, 1)[1].split('## 2.', 1)[0].strip(),
                           author='Working specification')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    for title, steps in FIGURES:
        web.append('<section class="figure"><h2>' + markup(title) + '</h2>' + ''.join('<div class="block">' + markup(s) + '</div>' for s in steps) + '</section>')
    HTML.write_text('<!doctype html><html lang="' + HTML_LANG + '"><meta charset="utf-8"><title>Patent specification</title><style>body{max-width:850px;margin:40px auto;font:16px/1.5 Georgia;color:#111}table{border-collapse:collapse;width:100%;font-size:13px}td{border:1px solid #555;padding:7px;overflow-wrap:anywhere}tr:first-child{background:#eee}pre{font-size:12px;white-space:pre-wrap}.figure{break-before:page;margin-top:50px}.block{border:1px solid #222;padding:14px;text-align:center;margin:12px}h2,h3{break-after:avoid}@media print{body{margin:0}h2{font-size:18px}}</style><body>' + '\n'.join(web) + '</body></html>', encoding='utf-8')
    abstract = text.split(ABSTRACT_SECTION, 1)[1].strip()
    assert 50 <= len(abstract.split()) <= ABSTRACT_MAX_WORDS, len(abstract.split())
    assert '{{' not in text
    print(f'Built {PDF}\nAbstract: {len(abstract.split())} words')


if __name__ == '__main__':
    main()
