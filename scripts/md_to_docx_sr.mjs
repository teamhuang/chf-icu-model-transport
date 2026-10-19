#!/usr/bin/env node
/**
 * md_to_docx_sr.mjs — Scientific Reports (Nature-style) Markdown -> .docx renderer.
 * Extends the docx-skill Node fallback with two things this manuscript needs:
 *   1. superscript in-text citations marked with «1-3» (rendered as superscript runs);
 *   2. images scaled by their true aspect ratio to a target width (no 480x320 distortion).
 * Supports: # headings, bold/italic/inline-code, pipe tables, ordered/unordered lists,
 * images ![alt](path), blockquotes, horizontal rules, links.
 * Usage: node md_to_docx_sr.mjs input.md output.docx [--title "..."]
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
let docx;
try { docx = require("docx"); }
catch {
  const { execSync } = require("node:child_process");
  const root = execSync("npm root -g").toString().trim();
  docx = require(path.join(root, "docx"));
}
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, LevelFormat } = docx;

const argv = process.argv.slice(2);
const positional = argv.filter(a => !a.startsWith("--"));
const getFlagVal = (n) => { const i = argv.indexOf(n); return i>=0 && argv[i+1] && !argv[i+1].startsWith("--") ? argv[i+1] : null; };
const inputPath = positional[0], outputPath = positional[1];
const docTitle = getFlagVal("--title");
if (!inputPath || !outputPath) { console.error("usage: node md_to_docx_sr.mjs in.md out.docx"); process.exit(2); }
const mdDir = path.dirname(path.resolve(inputPath));
const md = fs.readFileSync(inputPath, "utf8");

const FONT = "Times New Roman";           // SR accepted; serif body
const HEADFONT = "Arial";
const PAGE_WIDTH = 9026;                  // A4 usable with 1in margins (DXA)
const IMG_TARGET_PX = 560;                // ~5.83in wide max

// PNG intrinsic size from IHDR
function pngSize(p){ const b=fs.readFileSync(p); if(b.slice(1,4).toString("ascii")!=="PNG"&&b[1]!==0x50){} return {w:b.readUInt32BE(16),h:b.readUInt32BE(20)}; }
function jpegSize(p){ const b=fs.readFileSync(p); let i=2; while(i<b.length){ if(b[i]!==0xFF){i++;continue;} const m=b[i+1]; if(m>=0xC0&&m<=0xCF&&m!==0xC4&&m!==0xC8&&m!==0xCC){ return {w:b.readUInt16BE(i+7),h:b.readUInt16BE(i+5)};} i+=2+b.readUInt16BE(i+2);} return {w:1200,h:800}; }
function imgSize(p){ const e=path.extname(p).toLowerCase(); try{ return e===".jpg"||e===".jpeg"?jpegSize(p):pngSize(p);}catch{ return {w:1200,h:800}; } }

// inline parser: bold **, italic *, code `, link [](), superscript «»
function parseInline(text){
  const runs=[];
  const re=/(\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|_([^_]+)_|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\u00ab([^\u00bb]+)\u00bb)/g;
  let last=0,m;
  const push=(t,opts={})=>{ if(t) runs.push(new TextRun({text:t,font:FONT,...opts})); };
  while((m=re.exec(text))!==null){
    push(text.slice(last,m.index));
    if(m[2]||m[3]) push(m[2]||m[3],{bold:true});
    else if(m[4]||m[5]) push(m[4]||m[5],{italics:true});
    else if(m[6]) runs.push(new TextRun({text:m[6],font:"Consolas"}));
    else if(m[7]) push(`${m[7]} (${m[8]})`,{style:"Hyperlink"});
    else if(m[9]) runs.push(new TextRun({text:m[9],font:FONT,superScript:true}));
    last=re.lastIndex;
  }
  push(text.slice(last));
  return runs.length?runs:[new TextRun({text:"",font:FONT})];
}
const headingLevels=[HeadingLevel.HEADING_1,HeadingLevel.HEADING_2,HeadingLevel.HEADING_3,HeadingLevel.HEADING_4,HeadingLevel.HEADING_5,HeadingLevel.HEADING_6];
function makeTable(headerCells,bodyRows){
  const cols=headerCells.length, colW=Math.floor(PAGE_WIDTH/cols), colWidths=Array(cols).fill(colW);
  const border={style:BorderStyle.SINGLE,size:1,color:"BFBFBF"};
  const borders={top:border,bottom:border,left:border,right:border};
  const mkRow=(cells,header)=>new TableRow({children:cells.map((c,i)=>new TableCell({borders,
    width:{size:colWidths[i],type:WidthType.DXA},
    shading:header?{fill:"E8EEF7",type:ShadingType.CLEAR}:undefined,
    margins:{top:60,bottom:60,left:100,right:100},
    children:[new Paragraph({children:parseInline(c.trim())})]}))});
  return new Table({width:{size:PAGE_WIDTH,type:WidthType.DXA},columnWidths:colWidths,
    rows:[mkRow(headerCells,true),...bodyRows.map(r=>mkRow(r,false))]});
}
const lines=md.split(/\r?\n/); const children=[]; let i=0;
if(docTitle) children.push(new Paragraph({heading:HeadingLevel.TITLE,children:[new TextRun({text:docTitle,font:HEADFONT})]}));
while(i<lines.length){
  let line=lines[i];
  if(/^```/.test(line)){ i++; const buf=[]; while(i<lines.length&&!/^```/.test(lines[i]))buf.push(lines[i++]); i++;
    for(const code of buf) children.push(new Paragraph({shading:{type:ShadingType.CLEAR,fill:"F2F2F2"},children:[new TextRun({text:code||" ",font:"Consolas",size:20})]})); continue; }
  if(/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)){ children.push(new Paragraph({border:{bottom:{style:BorderStyle.SINGLE,size:6,color:"999999",space:1}},children:[new TextRun("")]})); i++; continue; }
  const h=line.match(/^(#{1,6})\s+(.*)$/);
  if(h){ const lvl=h[1].length-1; children.push(new Paragraph({heading:headingLevels[lvl],children:parseInline(h[2].trim())})); i++; continue; }
  if(/\|/.test(line)&&i+1<lines.length&&/^\s*\|?[-: |]+\|?\s*$/.test(lines[i+1])&&/-/.test(lines[i+1])){
    const splitRow=r=>r.replace(/^\s*\|/,"").replace(/\|\s*$/,"").split("|");
    const header=splitRow(line); i+=2; const body=[];
    while(i<lines.length&&/\|/.test(lines[i])&&lines[i].trim()!==""){body.push(splitRow(lines[i]));i++;}
    children.push(makeTable(header,body)); children.push(new Paragraph({children:[new TextRun("")]})); continue; }
  const img=line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
  if(img){ const alt=img[1]||"figure"; let raw=img[2].trim();
    let p=raw; if(!path.isAbsolute(p)){ const cand=[path.join(mdDir,raw), path.join(process.cwd(),raw)]; p=cand.find(c=>fs.existsSync(c))||cand[0]; }
    if(fs.existsSync(p)){ const {w,h}=imgSize(p); const W=IMG_TARGET_PX, H=Math.round(W*h/w);
      const ext=path.extname(p).slice(1).toLowerCase().replace("jpeg","jpg");
      children.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:120},
        children:[new ImageRun({type:{png:"png",jpg:"jpg",gif:"gif",bmp:"bmp",svg:"svg"}[ext]||"png",
          data:fs.readFileSync(p),transformation:{width:W,height:H},altText:{title:alt,description:alt,name:alt}})]}));
    } else children.push(new Paragraph({children:[new TextRun({text:`[missing image: ${p}]`,font:FONT,italics:true})]}));
    i++; continue; }
  if(/^>\s?/.test(line)){ const text=line.replace(/^>\s?/,""); children.push(new Paragraph({indent:{left:480},border:{left:{style:BorderStyle.SINGLE,size:12,color:"CCCCCC",space:8}},children:parseInline(text)})); i++; continue; }
  if(/^\s*[-*+]\s+/.test(line)){ children.push(new Paragraph({numbering:{reference:"ul",level:0},children:parseInline(line.replace(/^\s*[-*+]\s+/,""))})); i++; continue; }
  if(/^\s*\d+\.\s+/.test(line)){ children.push(new Paragraph({numbering:{reference:"ol",level:0},children:parseInline(line.replace(/^\s*\d+\.\s+/,""))})); i++; continue; }
  if(line.trim()===""){ i++; continue; }
  children.push(new Paragraph({spacing:{after:120,line:276,lineRule:"auto"},children:parseInline(line.trim())})); i++;
}
const doc=new Document({
  styles:{default:{document:{run:{font:FONT,size:22}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:30,bold:true,color:"1F3864",font:HEADFONT},paragraph:{spacing:{before:240,after:160},outlineLevel:0}},
      {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:26,bold:true,color:"1F3864",font:HEADFONT},paragraph:{spacing:{before:200,after:120},outlineLevel:1}},
      {id:"Heading3",name:"Heading 3",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:23,bold:true,color:"000000",font:HEADFONT},paragraph:{spacing:{before:160,after:100},outlineLevel:2}},
    ]},
  numbering:{config:[
    {reference:"ul",levels:[{level:0,format:LevelFormat.BULLET,text:"\u2022",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:720,hanging:360}}}}]},
    {reference:"ol",levels:[{level:0,format:LevelFormat.DECIMAL,text:"%1.",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:720,hanging:360}}}}]},
  ]},
  sections:[{properties:{page:{margin:{top:1440,right:1440,bottom:1440,left:1440}}},children}],
});
Packer.toBuffer(doc).then(buf=>{ fs.writeFileSync(outputPath,buf); console.log("OK wrote",outputPath); })
  .catch(e=>{ console.error("ERROR:",e.message); process.exit(1); });
