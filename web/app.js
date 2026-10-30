const $=id=>document.getElementById(id);
let models=[],result='',busy=false;
function count(){ $('counter').textContent=`${$('input-text').value.length.toLocaleString()} / 10,000`; }
function updateMetric(){
  const model=models.find(m=>m.id===$('model').value);
  $('f1').textContent=model?.dev_macro_f1 ? Number(model.dev_macro_f1).toFixed(4) : '—';
  $('accuracy').textContent=model?.dev_accuracy ? (100*Number(model.dev_accuracy)).toFixed(2)+'%' : '—';
}
function clearResult(){result='';$('output').replaceChildren();$('output').hidden=true;$('empty').hidden=false;$('copy').disabled=true;$('change-count').textContent='Ready when you are';$('error').hidden=true;}
$('input-text').addEventListener('input',()=>{count();clearResult();});
$('model').addEventListener('change',()=>{updateMetric();clearResult();});
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>{$('input-text').value=button.dataset.sample;count();clearResult();$('input-text').focus();}));
$('restore-form').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;busy=true;$('submit').disabled=true;$('model').disabled=true;$('submit').firstChild.textContent='Restoring… ';clearResult();
 try{
  const response=await fetch('/api/restore',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('input-text').value,model:$('model').value})});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Prediction failed.');
  result=data.output;
  for(const part of data.segments){const node=document.createElement(part.changed?'mark':'span');node.textContent=part.text;$('output').append(node);}
  $('empty').hidden=true;$('output').hidden=false;$('copy').disabled=false;$('change-count').textContent=`${data.changed_tokens} token${data.changed_tokens===1?'':'s'} changed · ${data.model_label}`;
 }catch(error){$('error').textContent=error.message;$('error').hidden=false;}
 finally{busy=false;$('submit').disabled=false;$('model').disabled=false;$('submit').firstChild.textContent='Restore capitalization ';}
});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(result);$('copy').textContent='Copied';setTimeout(()=>{$('copy').textContent='Copy text';},1500);}catch{$('error').textContent='Select the restored text and copy it manually.';$('error').hidden=false;}});
async function init(){
 count();try{const response=await fetch('/api/models');if(!response.ok)throw new Error('Could not load models.');const data=await response.json();models=data.models;$('model').replaceChildren();
 for(const model of models){const option=document.createElement('option');option.value=model.id;option.textContent=model.label+(model.available?'':' — not available');option.disabled=!model.available;$('model').append(option);}
 const available=models.find(m=>m.available);if(!available)throw new Error('No models are available.');$('model').value=available.id;$('model').disabled=false;$('submit').disabled=false;updateMetric();
 }catch(error){$('error').textContent=error.message+' Start the Python demo server and reload this page.';$('error').hidden=false;}
}
init();
