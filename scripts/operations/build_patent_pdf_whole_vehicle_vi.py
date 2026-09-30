"""Vietnamese companion specification with embedded fonts and translated graphics."""
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import String
from reportlab.graphics import renderSVG
from reportlab.platypus import PageBreak, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

import build_patent_pdf_en_revised as base
import build_patent_pdf_whole_vehicle as graphics


TRANSLATIONS = {
 'VEHICLE REGION COVERAGE (100)': 'PHẠM VI VÙNG KIỂM TRA TOÀN XE (100)',
 'Propulsion compartment (160)': 'Khoang hệ truyền động (160)',
 'Cabin / cargo (130)': 'Khoang hành khách / hàng (130)',
 'Exterior / body (120)': 'Ngoại thất / thân xe (120)',
 'Wheels / tires (140)': 'Bánh xe / lốp (140)',
 'Underbody / chassis (150)': 'Gầm / khung xe (150)',
 'Electrical / energy (170)': 'Điện / năng lượng (170)',
 'Region names and component positions are illustrative; configuration determines applicability.': 'Tên vùng và vị trí linh kiện là minh họa; cấu hình xác định tính áp dụng.',
 'ICE / HYBRID / EV | Passenger / commercial / bus / motorcycle profiles': 'Đốt trong / lai / điện | Xe con / thương mại / buýt / xe máy',
 'Acquisition (202)\nViews + session / vehicle ID': 'Thu nhận (202)\nẢnh + định danh phiên / xe',
 'Region routing (204)\nProfile + view quality': 'Định tuyến vùng (204)\nHồ sơ + chất lượng ảnh',
 'Context / local analysis (206)\nRegion-specific or shared model': 'Phân tích ngữ cảnh / cục bộ (206)\nMô hình theo vùng hoặc dùng chung',
 'Evidence resolution (208)\nWithin-view + optional cross-view': 'Xử lý bằng chứng (208)\nTrong ảnh + tùy chọn giữa ảnh',
 'Hierarchy / coverage (210)\nIdentity + applicable region state': 'Phân cấp / bao phủ (210)\nĐịnh danh + trạng thái vùng',
 'Assessment (212)\nFindings + uncertainty': 'Đánh giá (212)\nPhát hiện + bất định',
 'Output / recapture (214)\nReport or additional view': 'Đầu ra / chụp lại (214)\nBáo cáo hoặc ảnh bổ sung',
 'Dashed path: targeted recapture returns new evidence to the inspection session.': 'Nét đứt: chụp lại có mục tiêu đưa bằng chứng mới vào phiên kiểm tra.',
 'Vehicle (302)': 'Phương tiện (302)',
 'Exterior region (304)': 'Vùng ngoại thất (304)',
 'Wheel / tire region (304)': 'Vùng bánh / lốp (304)',
 'Propulsion region (304)': 'Vùng truyền động (304)',
 'Body system (306)': 'Hệ thống thân xe (306)',
 'Braking system (306)': 'Hệ thống phanh (306)',
 'Cooling system (306)': 'Hệ thống làm mát (306)',
 'Door surface (308)': 'Bề mặt cửa (308)',
 'Brake component (308)': 'Linh kiện phanh (308)',
 'Coolant line (308)': 'Đường làm mát (308)',
 'relation': 'quan hệ',
 'Evidence nodes (310): view ID, location, score, finding, review and coverage state': 'Nút bằng chứng (310): mã ảnh, vị trí, điểm, phát hiện, thẩm định và trạng thái bao phủ',
 'Solid: ownership tree. Dashed: shared-function / connected-to relationship (320).': 'Nét liền: cây thuộc về. Nét đứt: quan hệ chức năng / kết nối (320).',
 'Parent image / region view (400)': 'Ảnh gốc / góc nhìn vùng (400)',
 'Context branch (402)\nOverview / large targets': 'Nhánh ngữ cảnh (402)\nTổng quan / đối tượng lớn',
 'Local branches (404)\nTiles / selected detail': 'Nhánh cục bộ (404)\nÔ ảnh / chi tiết được chọn',
 'Common parent-image coordinates (406)\nResolve duplicates; preserve nested distinct targets': 'Tọa độ chung của ảnh gốc (406)\nXử lý trùng; giữ đối tượng khác nhau lồng nhau',
 'Optional cross-view association (408)\nRegion / side + identity evidence; retain ambiguity': 'Liên kết giữa ảnh tùy chọn (408)\nVùng / phía + bằng chứng định danh; giữ bất định',
 'Different views are linked semantically; they need not share a pixel coordinate frame.': 'Liên kết ảnh theo ngữ nghĩa; không bắt buộc chung hệ tọa độ điểm ảnh.',
 'Accepted image evidence (502)\nQuality + target visibility': 'Bằng chứng ảnh chấp nhận (502)\nChất lượng + nhìn thấy mục tiêu',
 'Vehicle applicability (504)\nProfile + inspection policy': 'Tính áp dụng theo xe (504)\nHồ sơ + chính sách kiểm tra',
 'Observation state (506)\nCoverage is separate from condition': 'Trạng thái quan sát (506)\nBao phủ tách biệt tình trạng',
 'Observed': 'Đã quan sát',
 'Partial / unresolved': 'Một phần / chưa rõ',
 'Not inspected': 'Chưa kiểm tra',
 'Report observed findings\nPreserve supporting evidence': 'Báo cáo phát hiện quan sát\nGiữ bằng chứng hỗ trợ',
 'Targeted capture (508)\nRegion + view + reason': 'Chụp có mục tiêu (508)\nVùng + góc nhìn + lý do',
 'Not applicable is a profile state, not a failed or missing component.': 'Không áp dụng là trạng thái hồ sơ, không phải linh kiện hỏng hoặc thiếu.',
 'Completion requires the configured coverage criteria for applicable regions.': 'Hoàn tất đòi hỏi đáp ứng tiêu chí bao phủ cấu hình cho các vùng áp dụng.',
 'Candidates (602)\nComponent / surface / region': 'Ứng viên (602)\nLinh kiện / bề mặt / vùng',
 'Reviewed references (604)\nClass + localization verdicts': 'Mẫu đã thẩm định (604)\nKết quả lớp + định vị',
 'Verdict model (606)\nClass score + separate acceptance gates': 'Mô hình từ thẩm định (606)\nĐiểm lớp + điều kiện chấp nhận riêng',
 'Admit': 'Chấp nhận',
 'Expert review': 'Chuyên gia thẩm định',
 'Exclude': 'Loại',
 'Eligible reviewed real support (608)\nFine identity retained; generic fallback when needed': 'Mẫu thực thẩm định đủ điều kiện (608)\nGiữ định danh chi tiết; dự phòng chung khi cần',
 'Versioned training mapping (610)\nPromotion threshold > retention threshold': 'Ánh xạ huấn luyện có phiên bản (610)\nNgưỡng thăng hạng > ngưỡng duy trì',
 'Localized visual finding (702)\nTarget + view + assessment state': 'Phát hiện có định vị (702)\nMục tiêu + ảnh + trạng thái',
 'Diagnostic context (720)\nDTC / sensor / service record': 'Ngữ cảnh chẩn đoán (720)\nDTC / cảm biến / bảo dưỡng',
 'Component-linked evidence (704)\nRetain uncertainty and contradictions': 'Bằng chứng gắn linh kiện (704)\nGiữ bất định và mâu thuẫn',
 'Inspection assessment (706)\nObserved / suspected / review required': 'Đánh giá kiểm tra (706)\nQuan sát / nghi ngờ / cần thẩm định',
 'Session A': 'Phiên A', 'Session B': 'Phiên B', 'Session C': 'Phiên C',
 'View provenance\nIdentity / condition': 'Nguồn ảnh\nĐịnh danh / tình trạng',
 'Longitudinal comparison (740): corresponding targets, changes, and replacement events.': 'So sánh qua thời gian (740): mục tiêu tương ứng, thay đổi và thay thế.',
 'Body / cabin assets (802)': 'Mô hình thân / khoang xe (802)',
 'Wheel / chassis assets (802)': 'Mô hình bánh / khung xe (802)',
 'Propulsion / energy assets (802)': 'Mô hình truyền động / năng lượng (802)',
 'Domain randomization (804)\nViewpoint / lighting / materials / occlusion': 'Ngẫu nhiên hóa miền (804)\nGóc nhìn / ánh sáng / vật liệu / che khuất',
 'Renderer + labels (806)\nRGB / masks / regions / configured conditions': 'Kết xuất + nhãn (806)\nRGB / mặt nạ / vùng / tình trạng cấu hình',
 'Synthetic provenance (808)\nExclude from real promotion counts': 'Nguồn tổng hợp (808)\nKhông tính vào mẫu thực thăng hạng',
 'Expert-reviewed real data (810)\nEligible class support': 'Dữ liệu thực đã thẩm định (810)\nMẫu lớp đủ điều kiện',
 'Training / real adaptation (812)\nDeploy region-linked visual models': 'Huấn luyện / thích nghi thực (812)\nTriển khai mô hình gắn theo vùng',
 'Offline synthesis; no mandatory online three-dimensional reconstruction.': 'Tổng hợp ngoại tuyến; không bắt buộc tái dựng ba chiều khi vận hành.',
}


def main():
    font_dir=Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('PatentVI',str(font_dir/'times.ttf')))
    pdfmetrics.registerFont(TTFont('PatentVI-Bold',str(font_dir/'timesbd.ttf')))
    pdfmetrics.registerFontFamily('PatentVI',normal='PatentVI',bold='PatentVI-Bold',
                                italic='PatentVI',boldItalic='PatentVI-Bold')
    for style in base.styles.byName.values():
        if getattr(style,'fontName',None)=='Times-Roman': style.fontName='PatentVI'
        elif getattr(style,'fontName',None)=='Times-Bold': style.fontName='PatentVI-Bold'
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('PatentVI',8)
        canvas.drawString(25*mm,12*mm,'Dự thảo toàn xe - 30 tháng 9 năm 2026')
        canvas.drawRightString(A4[0]-20*mm,12*mm,str(doc.page)); canvas.restoreState()
    base.footer=footer
    original_box=graphics.box
    original_note=graphics.note
    def label(d,x,y,text,size=10,anchor='middle',bold=False):
        translated=TRANSLATIONS.get(text,text)
        d.add(String(x,y,translated,fontName='PatentVI-Bold' if bold else 'PatentVI',
                     fontSize=size,textAnchor=anchor,fillColor=graphics.BLACK))
    def box(d,x,y,w,h,text,size=10,fill=graphics.LIGHT):
        translated=TRANSLATIONS.get(text,text)
        original_box(d,x,y,w,h,translated,size,fill)
    def note(d,y,text): original_note(d,y,TRANSLATIONS.get(text,text))
    graphics.label=label; graphics.box=box; graphics.note=note
    titles=[
        'HÌNH 1. Phạm vi vùng kiểm tra toàn xe',
        'HÌNH 2. Kiến trúc kiểm tra và vòng chụp lại',
        'HÌNH 3. Phân cấp, nút bằng chứng và đồ thị quan hệ',
        'HÌNH 4. Phân tích ngữ cảnh/cục bộ và liên kết ảnh',
        'HÌNH 5. Trạng thái bao phủ và chụp bổ sung có mục tiêu',
        'HÌNH 6. Thẩm định và phản hồi trạng thái lớp',
        'HÌNH 7. Bằng chứng tình trạng và so sánh qua thời gian',
        'HÌNH 8. Bộ huấn luyện tổng hợp theo vùng xe',
    ]
    fig_dir=base.OUT/'figures_whole_vehicle_vi'
    fig_dir.mkdir(exist_ok=True)
    drawings=[]
    for i,((_,make),title) in enumerate(zip(graphics.SPECS,titles),1):
        drawing=make(); renderSVG.drawToFile(drawing,str(fig_dir/f'fig{i}_whole_vehicle_vi.svg'))
        drawings.append((title,drawing))
    def pages():
        story=[]
        for title,drawing in drawings:
            story.extend([PageBreak(),Paragraph(title,base.styles['PatentH2']),Spacer(1,16),
                          drawing,Spacer(1,14),Paragraph(
                              'Sơ đồ phương án thực hiện. Tính áp dụng của vùng, trạng thái bằng chứng và quan hệ linh kiện được xác định trong phần mô tả. Hình vẽ không xác lập hiệu năng toàn xe đã đo.',
                              base.styles['BodyPatent'])])
        return story
    base.SOURCE=base.OUT/'patent_specification_whole_vehicle_inspection_vi.md'
    base.PDF=base.OUT/'patent_specification_whole_vehicle_inspection_vi.pdf'
    base.HTML=base.OUT/'patent_specification_whole_vehicle_inspection_vi.html'
    base.TITLE_SECTION='## 1. TÊN SÁNG CHẾ'
    base.ABSTRACT_SECTION='## 8. TÓM TẮT'
    # Vietnamese whitespace separates syllables, unlike English word counts.
    base.ABSTRACT_MAX_WORDS=350
    base.HTML_LANG='vi'; base.FIGURES=[]; base.drawing_pages=pages
    base.main()
    page=base.HTML.read_text(encoding='utf-8'); figures=[]
    for i,(title,_) in enumerate(drawings,1):
        svg=(fig_dir/f'fig{i}_whole_vehicle_vi.svg').read_text(encoding='utf-8')
        figures.append(f'<section class="figure"><h2>{title}</h2>'+svg[svg.index('<svg'):]+'</section>')
    base.HTML.write_text(page.replace('</body>','\n'.join(figures)+'</body>'),encoding='utf-8')
    print('Vietnamese text, 30 claims and 8 translated vector figures generated.')


if __name__=='__main__': main()
