"""Build a high-level whole-vehicle patent draft with original vector graphics."""
from pathlib import Path
import math
import textwrap

from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, Polygon, String
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from reportlab.platypus import PageBreak, Paragraph, Spacer

import build_patent_pdf_en_revised as base

OUT = base.OUT
FIG_DIR = OUT / 'figures_whole_vehicle'
BLACK = colors.black
WHITE = colors.white
LIGHT = colors.HexColor('#f4f4f4')
WIDTH = 460


def label(d, x, y, text, size=10, anchor='middle', bold=False):
    d.add(String(x, y, text, fontName='Times-Bold' if bold else 'Times-Roman',
                 fontSize=size, textAnchor=anchor, fillColor=BLACK))


def box(d, x, y, w, h, text, size=10, fill=LIGHT):
    d.add(Rect(x, y, w, h, strokeColor=BLACK, fillColor=fill, strokeWidth=0.9))
    lines = []
    for part in text.split('\n'):
        lines.extend(textwrap.wrap(part, width=max(13, int(w/(size*0.48)))) or [''])
    for i, line in enumerate(lines):
        label(d, x+w/2, y+h/2+(len(lines)-1)*6-i*12-3, line, size)


def arrow(d, x1, y1, x2, y2, dashed=False):
    d.add(Line(x1, y1, x2, y2, strokeWidth=0.9, strokeColor=BLACK,
               strokeDashArray=[4, 3] if dashed else None))
    a = math.atan2(y2-y1, x2-x1)
    tip = [(x2,y2), (x2-7*math.cos(a-0.45),y2-7*math.sin(a-0.45)),
           (x2-7*math.cos(a+0.45),y2-7*math.sin(a+0.45))]
    d.add(Polygon([v for point in tip for v in point], fillColor=BLACK, strokeColor=BLACK))


def note(d, y, text):
    for n, line in enumerate(textwrap.wrap(text, 85)):
        label(d, WIDTH/2, y-n*12, line, 9)


def fig1():
    d = Drawing(WIDTH, 480)
    # Schematic vehicle side elevation, with service-region indicators.
    body = [65,215, 82,266, 137,273, 173,318, 282,318, 330,279,
            391,265, 402,215]
    d.add(Polygon(body, fillColor=WHITE, strokeColor=BLACK, strokeWidth=1.8))
    d.add(Polygon([145,279,179,309,226,309,226,279], fillColor=LIGHT,strokeColor=BLACK))
    d.add(Polygon([237,309,279,309,316,279,237,279],fillColor=LIGHT,strokeColor=BLACK))
    d.add(Line(229,279,229,222,strokeColor=BLACK))
    d.add(Line(315,277,315,222,strokeColor=BLACK))
    for x in (122,340):
        d.add(Circle(x,215,29,fillColor=WHITE,strokeColor=BLACK,strokeWidth=1.6))
        d.add(Circle(x,215,15,fillColor=LIGHT,strokeColor=BLACK))
    d.add(Rect(180,223,105,17,fillColor=LIGHT,strokeColor=BLACK))
    d.add(Rect(75,235,50,22,fillColor=LIGHT,strokeColor=BLACK))
    label(d,230,443,'VEHICLE REGION COVERAGE (100)',12,bold=True)
    box(d,12,368,135,46,'Propulsion compartment (160)')
    box(d,163,368,135,46,'Cabin / cargo (130)')
    box(d,313,368,135,46,'Exterior / body (120)')
    arrow(d,79,368,97,258); arrow(d,230,368,235,300); arrow(d,382,368,367,273)
    box(d,12,103,135,46,'Wheels / tires (140)')
    box(d,163,103,135,46,'Underbody / chassis (150)')
    box(d,313,103,135,46,'Electrical / energy (170)')
    arrow(d,80,149,118,187); arrow(d,230,149,230,219); arrow(d,380,149,286,230)
    note(d,62,'Region names and component positions are illustrative; configuration determines applicability.')
    note(d,31,'ICE / HYBRID / EV | Passenger / commercial / bus / motorcycle profiles')
    return d


def fig2():
    d=Drawing(WIDTH,480)
    nodes=[(20,386,190,56,'Acquisition (202)\nViews + session / vehicle ID'),
           (250,386,190,56,'Region routing (204)\nProfile + view quality'),
           (250,275,190,56,'Context / local analysis (206)\nRegion-specific or shared model'),
           (20,275,190,56,'Evidence resolution (208)\nWithin-view + optional cross-view'),
           (20,160,190,56,'Hierarchy / coverage (210)\nIdentity + applicable region state'),
           (250,160,190,56,'Assessment (212)\nFindings + uncertainty'),
           (133,52,194,56,'Output / recapture (214)\nReport or additional view')]
    for args in nodes: box(d,*args)
    arrow(d,210,414,250,414); arrow(d,345,386,345,331)
    arrow(d,250,303,210,303); arrow(d,115,275,115,216)
    arrow(d,210,188,250,188); arrow(d,345,160,290,108)
    arrow(d,170,108,115,160,dashed=True)
    # A recapture loop returning to image acquisition.
    d.add(Line(133,80,5,80,strokeDashArray=[4,3],strokeColor=BLACK))
    d.add(Line(5,80,5,414,strokeDashArray=[4,3],strokeColor=BLACK))
    arrow(d,5,414,20,414,dashed=True)
    note(d,23,'Dashed path: targeted recapture returns new evidence to the inspection session.')
    return d


def fig3():
    d=Drawing(WIDTH,470)
    box(d,165,395,130,40,'Vehicle (302)')
    labels=['Exterior','Wheel / tire','Propulsion']
    for x,txt in zip([10,165,320],labels):
        box(d,x,304,130,45,txt+' region (304)')
        arrow(d,230,395,x+65,349)
    for x,txt in zip([10,165,320],['Body system','Braking system','Cooling system']):
        box(d,x,222,130,42,txt+' (306)'); arrow(d,x+65,304,x+65,264)
    for x,txt in zip([10,165,320],['Door surface','Brake component','Coolant line']):
        box(d,x,139,130,42,txt+' (308)'); arrow(d,x+65,222,x+65,181)
    # Dashed relation graph distinct from tree edges.
    arrow(d,320,160,295,160,dashed=True)
    label(d,307,185,'relation',8)
    box(d,44,38,372,53,'Evidence nodes (310): view ID, location, score, finding, review and coverage state')
    for x in (75,230,385): arrow(d,x,139,x,91)
    note(d,10,'Solid: ownership tree. Dashed: shared-function / connected-to relationship (320).')
    return d


def fig4():
    d=Drawing(WIDTH,480)
    box(d,95,414,270,40,'Parent image / region view (400)')
    arrow(d,175,414,107,369); arrow(d,285,414,352,369)
    box(d,15,305,180,64,'Context branch (402)\nOverview / large targets')
    box(d,265,305,180,64,'Local branches (404)\nTiles / selected detail')
    # Mini contextual image and tiled view graphic.
    d.add(Rect(62,232,87,51,strokeColor=BLACK,fillColor=WHITE))
    d.add(Rect(79,242,26,23,strokeColor=BLACK,fillColor=None))
    for x,y in [(306,257),(337,257),(306,232),(337,232)]:
        d.add(Rect(x,y,37,31,strokeColor=BLACK,fillColor=None))
    arrow(d,106,232,172,190); arrow(d,354,232,288,190)
    box(d,70,132,320,58,'Common parent-image coordinates (406)\nResolve duplicates; preserve nested distinct targets')
    arrow(d,230,132,230,93)
    box(d,70,36,320,57,'Optional cross-view association (408)\nRegion / side + identity evidence; retain ambiguity')
    note(d,11,'Different views are linked semantically; they need not share a pixel coordinate frame.')
    return d


def fig5():
    d=Drawing(WIDTH,480)
    box(d,20,396,190,50,'Accepted image evidence (502)\nQuality + target visibility')
    box(d,250,396,190,50,'Vehicle applicability (504)\nProfile + inspection policy')
    arrow(d,115,396,170,348); arrow(d,345,396,290,348)
    box(d,93,298,274,50,'Observation state (506)\nCoverage is separate from condition')
    for x,t in [(10,'Observed'),(165,'Partial / unresolved'),(320,'Not inspected')]:
        box(d,x,190,130,54,t); arrow(d,230,298,x+65,244)
    box(d,22,84,180,52,'Report observed findings\nPreserve supporting evidence')
    box(d,258,84,180,52,'Targeted capture (508)\nRegion + view + reason')
    arrow(d,75,190,112,136); arrow(d,230,190,348,136); arrow(d,385,190,348,136)
    arrow(d,438,110,452,110,dashed=True)
    d.add(Line(452,110,452,423,strokeDashArray=[4,3],strokeColor=BLACK))
    d.add(Line(452,423,452,466,strokeDashArray=[4,3],strokeColor=BLACK))
    d.add(Line(452,466,115,466,strokeDashArray=[4,3],strokeColor=BLACK))
    arrow(d,115,466,115,446,dashed=True)
    note(d,48,'Not applicable is a profile state, not a failed or missing component.')
    note(d,22,'Completion requires the configured coverage criteria for applicable regions.')
    return d


def fig6():
    d=Drawing(WIDTH,480)
    box(d,15,398,185,51,'Candidates (602)\nComponent / surface / region')
    box(d,260,398,185,51,'Reviewed references (604)\nClass + localization verdicts')
    arrow(d,107,398,175,355); arrow(d,352,398,285,355)
    box(d,103,302,254,53,'Verdict model (606)\nClass score + separate acceptance gates')
    for x,t in [(10,'Admit'),(165,'Expert review'),(320,'Exclude')]:
        box(d,x,207,130,44,t); arrow(d,230,302,x+65,251)
    box(d,74,110,312,52,'Eligible reviewed real support (608)\nFine identity retained; generic fallback when needed')
    arrow(d,230,207,230,162)
    box(d,74,30,312,46,'Versioned training mapping (610)\nPromotion threshold > retention threshold')
    arrow(d,230,110,230,76)
    d.add(Line(386,53,452,53,strokeDashArray=[4,3],strokeColor=BLACK))
    d.add(Line(452,53,452,424,strokeDashArray=[4,3],strokeColor=BLACK))
    arrow(d,452,424,445,424,dashed=True)
    return d


def fig7():
    d=Drawing(WIDTH,480)
    box(d,18,390,195,54,'Localized visual finding (702)\nTarget + view + assessment state')
    box(d,247,390,195,54,'Diagnostic context (720)\nDTC / sensor / service record')
    arrow(d,115,390,174,343); arrow(d,345,390,286,343)
    box(d,94,287,272,56,'Component-linked evidence (704)\nRetain uncertainty and contradictions')
    arrow(d,230,287,230,249)
    box(d,94,197,272,52,'Inspection assessment (706)\nObserved / suspected / review required')
    # Timeline anchors and view cards.
    arrow(d,38,136,419,136)
    for x,txt in [(90,'Session A'),(230,'Session B'),(370,'Session C')]:
        d.add(Circle(x,136,5,fillColor=WHITE,strokeColor=BLACK))
        label(d,x,155,txt,10,bold=True)
        box(d,x-57,54,114,48,'View provenance\nIdentity / condition')
        arrow(d,x,131,x,102)
    note(d,26,'Longitudinal comparison (740): corresponding targets, changes, and replacement events.')
    return d


def fig8():
    d=Drawing(WIDTH,480)
    for x,t in [(10,'Body / cabin'),(165,'Wheel / chassis'),(320,'Propulsion / energy')]:
        box(d,x,395,130,49,t+' assets (802)')
        arrow(d,x+65,395,230,350)
    box(d,84,294,292,56,'Domain randomization (804)\nViewpoint / lighting / materials / occlusion')
    arrow(d,230,294,230,259)
    box(d,84,210,292,49,'Renderer + labels (806)\nRGB / masks / regions / configured conditions')
    arrow(d,230,210,230,171)
    box(d,20,119,190,52,'Synthetic provenance (808)\nExclude from real promotion counts')
    box(d,250,119,190,52,'Expert-reviewed real data (810)\nEligible class support')
    arrow(d,210,145,250,145)
    arrow(d,115,119,175,77); arrow(d,345,119,285,77)
    box(d,95,30,270,47,'Training / real adaptation (812)\nDeploy region-linked visual models')
    note(d,9,'Offline synthesis; no mandatory online three-dimensional reconstruction.')
    return d


SPECS = [
    ('FIG. 1. Vehicle-wide region coverage',fig1),
    ('FIG. 2. Inspection architecture and recapture loop',fig2),
    ('FIG. 3. Hierarchy, evidence nodes and relationship graph',fig3),
    ('FIG. 4. Contextual/local analysis and view association',fig4),
    ('FIG. 5. Coverage states and targeted additional capture',fig5),
    ('FIG. 6. Verdict review and class-state feedback',fig6),
    ('FIG. 7. Condition evidence and longitudinal comparison',fig7),
    ('FIG. 8. Vehicle-region synthetic training engine',fig8),
]


def main():
    FIG_DIR.mkdir(exist_ok=True)
    drawings = []
    for i,(title,make) in enumerate(SPECS,1):
        drawing=make()
        renderSVG.drawToFile(drawing,str(FIG_DIR/f'fig{i}_whole_vehicle.svg'))
        drawings.append((title,drawing))
    def pages():
        story=[]
        for title,drawing in drawings:
            story.extend([PageBreak(),Paragraph(title,base.styles['PatentH2']),
                          Spacer(1,16),drawing,Spacer(1,14),
                          Paragraph('Schematic embodiment. Region applicability, evidence state and component relationships are defined in the description. The drawing does not establish measured whole-vehicle performance.',base.styles['BodyPatent'])])
        return story
    base.SOURCE=OUT/'patent_specification_whole_vehicle_inspection_en.md'
    base.PDF=OUT/'patent_specification_whole_vehicle_inspection_en.pdf'
    base.HTML=OUT/'patent_specification_whole_vehicle_inspection_en.html'
    base.FIGURES=[]
    base.drawing_pages=pages
    base.main()
    page=base.HTML.read_text(encoding='utf-8')
    figures=[]
    for i,(title,_) in enumerate(drawings,1):
        svg=(FIG_DIR/f'fig{i}_whole_vehicle.svg').read_text(encoding='utf-8')
        svg=svg[svg.index('<svg'):]
        figures.append(f'<section class="figure"><h2>{title}</h2>{svg}</section>')
    base.HTML.write_text(page.replace('</body>', '\n'.join(figures)+'</body>'),encoding='utf-8')
    print(f'Generated {len(drawings)} original vector figures in {FIG_DIR}')


if __name__=='__main__':
    main()
