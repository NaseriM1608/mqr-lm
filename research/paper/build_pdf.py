"""Render the editable paper into a two-column conference-style draft."""
from pathlib import Path
import hashlib
import html
import json
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                               PageBreak, FrameBreak, Table, TableStyle, Image, KeepTogether, Flowable)
from pypdf import PdfReader, PdfWriter, Transformation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT=HERE/'paper-rebuilt.pdf'
WIDTH,HEIGHT=A4
LEFT,RIGHT,TOP,BOTTOM=45,45,41,43
GAP=18
CW=(WIDTH-LEFT-RIGHT-GAP)/2

STYLES={
 'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.7,
                      alignment=TA_JUSTIFY,spaceAfter=5.3,splitLongWords=False),
 'section':ParagraphStyle('section',fontName='Times-Bold',fontSize=11,leading=13,
                         spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'subsection':ParagraphStyle('subsection',fontName='Times-Bold',fontSize=10,leading=11.8,
                            spaceBefore=7,spaceAfter=4,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.5,leading=10,
                         alignment=TA_LEFT,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=7.3,leading=8.6),
 'headcell':ParagraphStyle('headcell',fontName='Times-Bold',fontSize=7.1,leading=8.4),
 'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=9,leading=10.5,spaceAfter=7,
                     splitLongWords=True),
}
STYLES['appendixbody']=ParagraphStyle('appendixbody',parent=STYLES['body'],fontSize=9,leading=10.5,spaceAfter=4)
STYLES['appendixcaption']=ParagraphStyle('appendixcaption',parent=STYLES['caption'],fontSize=7.8,leading=9,spaceAfter=6)

def mark_up(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    for literal,formatted in [('R_50','R<sub>50</sub>'),('Z_j','Z<sub>j</sub>'),('a_jv','a<sub>jv</sub>')]:
        s=s.replace(literal,formatted)
    s=re.sub(r'(https://\S+)',lambda m:'<link href="'+m[1]+'" color="#234b6c">'+m[1]+'</link>',s)
    return s

class Marker(Flowable):
    def __init__(self,key):super().__init__();self.key=key;self.width=0;self.height=0
    def draw(self):setattr(self.canv._doctemplate,self.key,self.canv.getPageNumber())

class Equation(Flowable):
    """Reserve display space, number it, and record vector-overlay placement."""
    def __init__(self,path,number):
        super().__init__()
        self.path=path
        self.number=number
        page=PdfReader(path).pages[0]
        source_w,source_h=float(page.mediabox.width),float(page.mediabox.height)
        self.scale=min(1.0,(CW-26)/source_w)
        self.equation_width=source_w*self.scale
        self.equation_height=source_h*self.scale
        self.width=CW
        self.height=self.equation_height+4
    def draw(self):
        x=(CW-22-self.equation_width)/2
        y=2
        absolute_x,absolute_y=self.canv.absolutePosition(x,y)
        self.canv._doctemplate.equation_placements.append({
            'page':self.canv.getPageNumber(),'x':absolute_x,'y':absolute_y,
            'scale':self.scale,'file':str(self.path),'number':self.number})
        self.canv.setFont('Times-Roman',9)
        self.canv.drawRightString(CW,y+self.equation_height/2-3,f'({self.number})')

class Paper(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(OUTPUT),pagesize=A4,leftMargin=LEFT,rightMargin=RIGHT,
                         topMargin=TOP,bottomMargin=BOTTOM,
                         title='Query Diversity Without a Likelihood Advantage: A Controlled Study of Multi-Query Vocabulary Heads',
                         author='Research manuscript draft; author metadata pending researcher verification')
        self.equation_placements=[]
        def frames(top):
            return [Frame(x,BOTTOM,CW,HEIGHT-top-BOTTOM,leftPadding=0,rightPadding=0,
                          topPadding=0,bottomPadding=0,id=str(x)) for x in (LEFT,LEFT+CW+GAP)]
        self.addPageTemplates([
            PageTemplate(id='first',frames=frames(126),onPage=self.decorate,autoNextPageTemplate='normal'),
            PageTemplate(id='normal',frames=frames(TOP),onPage=self.decorate),
        ])
    def decorate(self,c,doc):
        c.setFillColor(colors.black)
        if doc.page==1:
            c.setFont('Times-Bold',15)
            for y,line in zip([HEIGHT-49,HEIGHT-68,HEIGHT-87],
                              ['Query Diversity Without a Likelihood Advantage:',
                               'A Controlled Study of Multi-Query Vocabulary Heads',
                               ]):
                c.drawCentredString(WIDTH/2,y,line)
            c.setFont('Times-Roman',9)
            c.drawCentredString(WIDTH/2,HEIGHT-108,'Research manuscript draft | Revised 28 October 2026')
        else:
            c.setFont('Times-Italic',8)
            c.drawString(LEFT,HEIGHT-25,'Multi-Query Vocabulary Heads: Controlled Empirical Study')
            c.drawRightString(WIDTH-RIGHT,HEIGHT-25,'Manuscript draft')
        c.setFont('Times-Roman',9);c.drawCentredString(WIDTH/2,24,str(doc.page))

def make_image(path,max_height=None,math=False):
    with PILImage.open(path) as im:w,h=im.size
    if math:
        # Equation PNGs were rendered at 500 dpi with a 10pt source font.
        target_w=min(CW,w/500*72)
    else:target_w=CW
    target_h=target_w*h/w
    if max_height and target_h>max_height:
        target_w*=max_height/target_h;target_h=max_height
    obj=Image(str(path),width=target_w,height=target_h)
    obj.hAlign='CENTER'
    return obj

def build():
    text=(HERE/'manuscript.md').read_text(encoding='utf-8')
    lines=text.splitlines();story=[];i=0;eq=0;references=False;appendix=False
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('# ') or line.startswith('Research manuscript draft'):
            i+=1;continue
        if line=='<!-- MAIN_END -->':
            story.append(Marker('main_last_page'));i+=1;continue
        if line=='<!-- APPENDIX_START -->':
            story.extend([PageBreak(),Marker('appendix_first_page')]);appendix=True;references=False;i+=1;continue
        if line.startswith('## '):
            title=line[3:]
            if title=='References':
                references=True
                story.append(FrameBreak())
            story.append(Paragraph(mark_up(title),STYLES['section']));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(mark_up(line[4:]),STYLES['subsection']));i+=1;continue
        if line.startswith('$$ '):
            story.append(Equation(HERE/'assets'/f'equation_{eq:02d}.pdf',eq+1))
            story.append(Spacer(1,4));eq+=1;i+=1;continue
        if line.startswith('!['):
            match=re.fullmatch(r'!\[(.+)\]\((.+)\)',line)
            story.append(KeepTogether([make_image(HERE/match[2]),Spacer(1,4),
                                      Paragraph(mark_up(match[1]),STYLES['caption'])]))
            i+=1;continue
        if line.startswith('|'):
            table_lines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip());i+=1
            parsed=[[s.strip() for s in row.strip('|').split('|')] for row in table_lines]
            parsed=[row for row in parsed if not all(re.fullmatch(r'[:\- ]+',s) for s in row)]
            count=len(parsed[0]);head=parsed[0]
            if head[0]=='Head':ratios=[.19,.13,.13,.27,.28]
            elif head[0]=='Adjusted-MQR diagnostic, B128':ratios=[.61,.39]
            elif head[0]=='Retained artifact':ratios=[.56,.44]
            elif head[0]=='Language' and count==6:ratios=[.16,.168,.168,.168,.168,.168]
            elif count==5 and head[1]=='B':ratios=[.23,.08,.23,.23,.23]
            elif count==5 and 'Exact decode' in head[1]:ratios=[.20,.17,.19,.22,.22]
            elif count==4:ratios=[.30,.23,.235,.235]
            else:ratios=[1/count]*count
            cellstyle=STYLES['cell']
            if count>=5:cellstyle=ParagraphStyle('smallcell',parent=cellstyle,fontSize=6.7,leading=7.8)
            content=[[Paragraph(mark_up(cell),STYLES['headcell'] if n==0 else cellstyle)
                      for cell in row] for n,row in enumerate(parsed)]
            table=Table(content,colWidths=[CW*r for r in ratios],hAlign='LEFT',repeatRows=1)
            table.setStyle(TableStyle([
                ('LINEABOVE',(0,0),(-1,0),.7,colors.black),
                ('LINEBELOW',(0,0),(-1,0),.45,colors.black),
                ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),
                ('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),
                ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
                ('VALIGN',(0,0),(-1,-1),'TOP')]))
            while i<len(lines) and not lines[i].strip():i+=1
            elements=[table,Spacer(1,4)]
            if i<len(lines) and re.match(r'Table [A-Z]?\d',lines[i]):
                elements.append(Paragraph(mark_up(lines[i]),STYLES['appendixcaption' if appendix else 'caption']));i+=1
            story.append(KeepTogether(elements));continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','$$','<!--')):
            paragraph.append(lines[i].strip());i+=1
        story.append(Paragraph(mark_up(' '.join(paragraph)),STYLES['ref' if references else 'appendixbody' if appendix else 'body']))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    doc=Paper();doc.build(story)
    # Preserve the source PDF equations' embedded vector glyphs rather than
    # rasterizing them into page images. The scientific source remains intact.
    base=PdfReader(OUTPUT)
    writer=PdfWriter()
    for page in base.pages:writer.add_page(page)
    if base.metadata:writer.add_metadata(base.metadata)
    for placement in doc.equation_placements:
        equation_page=PdfReader(placement['file']).pages[0]
        transform=Transformation().scale(placement['scale']).translate(placement['x'],placement['y'])
        writer.pages[placement['page']-1].merge_transformed_page(equation_page,transform)
    temporary=OUTPUT.with_suffix('.vector-tmp.pdf')
    with temporary.open('wb') as f:writer.write(f)
    temporary.replace(OUTPUT)
    reader=PdfReader(OUTPUT)
    main_pages=doc.main_last_page
    result={'pdf':str(OUTPUT),'pages':len(reader.pages),'main_text_pages':main_pages,
            'appendix_first_page':doc.appendix_first_page,
            'main_text_words':len(text.split('<!-- MAIN_END -->')[0].split()),
            'equations':eq,'figures':len(re.findall(r'^!\[',text,re.M)),
            'equation_font':'Computer Modern','equation_rendering':'Embedded vector glyphs',
            'equation_numbers':[p['number'] for p in doc.equation_placements],
            'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            'page_text_characters':[len(p.extract_text()) for p in reader.pages]}
    OUTPUT.with_suffix('.layout.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':build()
