"""Self-contained HTML/SVG report; no web service, CDN or plotting dependency."""

import json
from pathlib import Path

from openwam.evaluation.nero_metrics import DIM_NAMES


def write_report(output, arrays, manifest, metrics):
    payload = {
        k: arrays[k].tolist() for k in ("gt", "pred", "native_gt", "native_pred", "state", "times", "episode", "anchor")
    }
    payload.update(names=DIM_NAMES, manifest=manifest, metrics=metrics)
    # Task text is untrusted dataset content; never interpolate it as HTML/script.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    Path(output, "report.html").write_text(TEMPLATE.replace("__PAYLOAD__", data))


TEMPLATE = r"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>Nero 开环评估</title>
<style>
body{font:15px system-ui,sans-serif;background:#f5f6f8;color:#1e293b;margin:24px auto;max-width:1440px;padding:0 20px}
h1{font-size:26px}p{line-height:1.7}select,button{padding:7px;margin:4px}label{margin-right:15px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:14px}
.card{background:white;padding:14px;border:1px solid #dce1e7;border-radius:8px}
svg{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}
td,th{text-align:left;padding:8px;border:1px solid #dce1e7}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.legend{color:#64748b}img{width:320px;max-width:100%}.error{color:#b91c1c}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
<h1>Nero episode 开环评估</h1>
<p>真实当前观察 → 未来 32 步预测。训练集拟合评估，不代表闭环任务成功率。<br>
<span style="color:#2563eb">蓝：GT</span> · <span style="color:#e35d30">橙：Predict</span> · 灰色虚线：保持当前状态。</p>
<p id="identity"></p>
<div class="card"><label>Episode <select id="episode"></select></label>
<label>视图 <select id="view"><option value="window">单窗口，完整 32 步</option><option value="episode">Episode 拼接（重叠处用最新锚点）</option></select></label>
<label>锚点 <select id="window"></select></label>
<label>动作空间 <select id="space"><option value="physical">绝对 EEF（部署输出）</option><option value="native">训练动作空间（处理前原始预测）</option></select></label>
<button id="save">导出当前 20 维 SVG</button></div>
<p id="context"></p><div class="grid"><div class="card"><img id="image"><p>上：头部；下左：左腕；下右：右腕</p></div>
<div class="card"><pre id="prompt"></pre></div></div>
<p>位置与夹爪曲线使用 mm；rot6D 为无单位的矩阵列分量。横轴为目标帧的 episode 时间（秒）。
窗口起点竖线标注重新预测；重叠预测完整保存在 NPZ 中。原始 CSV 使用 m。</p>
<div id="charts" class="grid"></div>
<h2>物理误差（选中 Episode，所有窗口×horizon）</h2><div id="summary"></div>
<h2>误差随预测距离变化</h2><p>蓝：保持当前状态基线的误差；橙：模型误差。横轴为 horizon（1…32）。</p><div id="horizons" class="grid"></div>
<details><summary>完整指标和原始预测诊断</summary><pre id="metrics"></pre></details>
<p><a href="predictions.csv">逐维 CSV</a> · <a href="predictions.npz">完整 NPZ</a> · <a href="metrics.json">指标 JSON</a> · <a href="manifest.json">运行记录</a></p>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const episodes=[...new Set(D.episode)], fmt=x=>x==null?'invalid':Number(x).toPrecision(4);
for(const e of episodes){const o=new Option(String(e),String(e));$('episode').add(o)}
$('identity').textContent=`Checkpoint: ${D.manifest.checkpoint} | mode=${D.manifest.action_mode} | seed=${D.manifest.seed} | denoise=${D.manifest.denoise_steps}`;
$('metrics').textContent=JSON.stringify(D.metrics,null,2);
function svgPlot(title,x,ys,breaks=[]){
 const W=640,H=220,L=65,R=16,T=30,B=38;
 const values=ys.flat().filter(Number.isFinite); let lo=Math.min(...values),hi=Math.max(...values);
 const pad=Math.max((hi-lo)*.07,1e-6);lo-=pad;hi+=pad;
 const xmin=x[0],xmax=x[x.length-1], X=v=>L+(v-xmin)/(xmax-xmin||1)*(W-L-R),Y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"><rect width="100%" height="100%" fill="white"/><text x="${L}" y="18" font-size="13">${title}</text>`;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=Y(v);s+=`<path d="M${L} ${y}H${W-R}" stroke="#e2e8f0"/><text x="4" y="${y+4}" font-size="10">${fmt(v)}</text>`}
 for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4;s+=`<text x="${X(v)-10}" y="${H-15}" font-size="10">${fmt(v)}</text>`}
 for(const b of breaks)s+=`<path d="M${X(b)} ${T}V${H-B}" stroke="#ccd2db" stroke-dasharray="3 4"/>`;
 ys.forEach((y,k)=>{let active=false,path='';for(let i=0;i<y.length;i++){if(!Number.isFinite(y[i])){active=false;continue}path+=`${active?'L':'M'}${X(x[i]).toFixed(2)} ${Y(y[i]).toFixed(2)} `;active=true}
 s+=`<path d="${path}" fill="none" stroke="${['#2563eb','#e35d30','#94a3b8'][k]}" stroke-width="1.5" ${k===2?'stroke-dasharray="4 3"':''}/>`});return s+'</svg>';
}
function fillWindows(){const ids=D.episode.map((e,i)=>e===+$('episode').value?i:-1).filter(i=>i>=0);$('window').replaceChildren();for(const i of ids)$('window').add(new Option(`t=${D.anchor[i]}`,i));render()}
function render(){
 const w=+$('window').value,e=+$('episode').value, native=$('space').value==='native',full=$('view').value==='episode';
 const gt=native?D.native_gt:D.gt,pred=native?D.native_pred:D.pred;
 let pairs=[];const ids=D.episode.map((x,i)=>x===e?i:-1).filter(i=>i>=0);
 if(full){const rows=new Map();for(const i of ids)for(let h=0;h<32;h++)rows.set(D.anchor[i]+h+1,[i,h]);pairs=[...rows.entries()].sort((a,b)=>a[0]-b[0]).map(a=>a[1])}
 else pairs=Array.from({length:32},(_,h)=>[w,h]);
 const x=pairs.map(([i,h])=>D.times[i][h]);
 const breaks=full?pairs.filter((p,j)=>!j||pairs[j-1][0]!==p[0]).map(([i,h])=>D.times[i][h]):[];
 $('context').textContent=`Episode ${e} | 显示 ${full?'拼接预测':`t=${D.anchor[w]} → t+1…t+32`} | ${native?'原始训练动作；相对模型的各 chunk 锚点不同':'绝对 body / gripper_flange'}`;
 $('image').src=D.manifest.windows[w].image;$('prompt').textContent=D.manifest.windows[w].prompt;
 $('charts').replaceChildren();
 for(let d=0;d<20;d++){
  const scale=(d%10<3||d%10===9)?1000:1;
  const ys=[gt,pred].map(a=>pairs.map(([i,h])=>a[i][h][d]*scale));
  if(!native)ys.push(pairs.map(([i])=>D.state[i][d]*scale));
  // Sparse explicit anchors must not be connected across unevaluated intervals.
  const xx=[], yy=ys.map(()=>[]);pairs.forEach(([i,h],j)=>{if(j&&D.anchor[i]+h+1>D.anchor[pairs[j-1][0]]+pairs[j-1][1]+2){xx.push((x[j-1]+x[j])/2);yy.forEach(a=>a.push(null))}xx.push(x[j]);yy.forEach((a,k)=>a.push(ys[k][j]))});
  const card=document.createElement('div');card.className='card';card.innerHTML=svgPlot(D.names[d].replace('_m',' [mm]'),xx,yy,breaks);$('charts').append(card)
 }
 const m=D.metrics.by_episode[String(e)];let table='<table><tr><th>臂</th><th>指标</th><th>模型均值</th><th>保持当前均值</th><th>模型 P95</th></tr>';
 for(const arm of ['left','right'])for(const key of ['position_mm','rotation_deg','gripper_mm']){const a=m.prediction.arms[arm][key],b=m.hold_current.arms[arm][key];table+=`<tr><td>${arm}</td><td>${key}</td><td>${fmt(a.mean)}</td><td>${fmt(b.mean)}</td><td>${fmt(a.p95)}</td></tr>`}
 $('summary').innerHTML=table+'</table>';$('horizons').replaceChildren();
 for(const arm of ['left','right'])for(const key of ['position_mm','rotation_deg','gripper_mm']){const node=document.createElement('div');node.className='card';node.innerHTML=svgPlot(`${arm} ${key} — horizon 1…32`,Array.from({length:32},(_,i)=>i+1),[m.hold_current.arms[arm][key+'_by_horizon'],m.prediction.arms[arm][key+'_by_horizon']]);$('horizons').append(node)}
}
$('episode').onchange=fillWindows;for(const id of ['window','view','space'])$(id).onchange=render;
$('save').onclick=()=>{const nodes=[...$('charts').querySelectorAll('svg')];const body=nodes.map((s,i)=>`<g transform="translate(${i%2*640} ${Math.floor(i/2)*220})">${s.innerHTML}</g>`).join('');const blob=new Blob([`<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="2200">${body}</svg>`],{type:'image/svg+xml'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='eef20_gt_predict.svg';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
fillWindows();
</script></html>"""
