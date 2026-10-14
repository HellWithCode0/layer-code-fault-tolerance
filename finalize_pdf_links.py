"""Export portable PDF links and a combined reading copy. Requires PyMuPDF."""
from pathlib import Path
import argparse, json
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent
NAMES={n:f'Katoch_FINAL_{n.upper()}_2026-10-14_v4.pdf' for n in ['manuscript','supplement']}
COMBINED='Katoch_FINAL_COMBINED_2026-10-14_v4.pdf'

def destination(doc, names, key):
    d=names[key]
    assert 0 <= d['page'] < len(doc), (key,d)
    # Named destinations expose PDF bottom-left coordinates. Link API uses
    # PyMuPDF top-left coordinates, so transform before inserting the link.
    pt=fitz.Point(*d.get('to',(0,doc[d['page']].rect.height))) * doc[d['page']].transformation_matrix
    return d['page'],pt

def main(out):
    out.mkdir(exist_ok=True,parents=True)
    docs={n:fitz.open(ROOT/(n+'.pdf')) for n in NAMES}
    names={n:d.resolve_names() for n,d in docs.items()}
    records={};report={}
    for name,doc in docs.items():
        records[name]=[]
        counts={'internal':0,'supplement':0,'web':0}
        for page in doc:
            page_records=[]
            for old in list(page.get_links()):
                base={'from':old['from']}
                if old.get('nameddest'):
                    p,pt=destination(doc,names[name],old['nameddest'])
                    new={**base,'kind':fitz.LINK_GOTO,'page':p,'to':pt,'zoom':0}
                    counts['internal']+=1
                elif old.get('file'):
                    filename,_,target=old['file'].partition('#nameddest=')
                    assert filename=='supplement.pdf' and target,old
                    p,pt=destination(docs['supplement'],names['supplement'],target)
                    new={**base,'kind':fitz.LINK_GOTOR,'file':NAMES['supplement'],'page':p,'to':pt,'zoom':0}
                    counts['supplement']+=1
                elif old.get('uri'):
                    new={**base,'kind':fitz.LINK_URI,'uri':old['uri']}
                    counts['web']+=1
                elif old['kind']==fitz.LINK_GOTO:
                    new={**base,'kind':fitz.LINK_GOTO,'page':old['page'],'to':old.get('to',fitz.Point(0,0))}
                    counts['internal']+=1
                else:
                    raise ValueError(f'Unrecognized link: {old}')
                page.delete_link(old)
                page.insert_link(new)
                page_records.append(new)
            records[name].append(page_records)
        path=out/NAMES[name]
        doc.save(path,garbage=4,deflate=True)
        report[name]=counts

    merged=fitz.open()
    offsets={'manuscript':0,'supplement':len(docs['manuscript'])}
    # Recreate links explicitly: a named destination in one PDF can have the
    # same name as a different destination in the other PDF.
    for doc in docs.values():
        merged.insert_pdf(doc,links=False)
    for name,doc in docs.items():
        for i,links in enumerate(records[name]):
            page=merged[offsets[name]+i]
            for old in links:
                new=dict(old)
                if old['kind']==fitz.LINK_GOTOR:
                    new.pop('file')
                    new['kind']=fitz.LINK_GOTO
                    new['page']=offsets['supplement']+old['page']
                elif old['kind']==fitz.LINK_GOTO:
                    new['page']=offsets[name]+old['page']
                page.insert_link(new)
    merged.set_toc([[1,'Manuscript',1],[1,'Supplemental Material',offsets['supplement']+1]])
    merged.set_metadata({'title':'Universal Fault-Tolerant Computation on Three-Dimensional Layer Codes at Optimal Distance: manuscript and supplement','author':'Aryaman Katoch'})
    merged.save(out/COMBINED,garbage=4,deflate=True)

    for filename in [*NAMES.values(),COMBINED]:
        d=fitz.open(out/filename)
        for page in d:
            for link in page.get_links():
                assert link['kind'] in [fitz.LINK_GOTO,fitz.LINK_GOTOR,fitz.LINK_URI],link
                if link['kind']==fitz.LINK_GOTO:assert 0<=link['page']<len(d)
                if link['kind']==fitz.LINK_GOTOR:
                    target=out/link['file']
                    assert target.exists(),target
                    with fitz.open(target) as remote:assert 0<=link['page']<len(remote)
        # Link changes must not alter visible page contents.
        if filename!=COMBINED:
            name=next(k for k,v in NAMES.items() if v==filename)
            assert len(d)==len(docs[name])
            for p,q in zip(d,docs[name]):
                assert p.get_pixmap().samples==q.get_pixmap().samples
    (ROOT/'revision_link_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',type=Path,default=ROOT/'linked_pdfs')
    main(ap.parse_args().out_dir)
