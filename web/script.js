document.addEventListener('DOMContentLoaded',()=>{
  const btn=document.getElementById('predict-btn');
  const input=document.getElementById('sample-input');
  const out=document.getElementById('prediction');
  const openResultsBtn = document.getElementById('open-results');
  if(openResultsBtn){
    openResultsBtn.addEventListener('click', (e)=>{ e && e.preventDefault && e.preventDefault(); window.location.replace('results.html'); });
  }
  btn.addEventListener('click',async ()=>{
    const text=input.value.trim();
    if(!text){out.textContent='Please enter a headline.';return}
    out.textContent='Analysing...';
    out.style.opacity=0.6;
    try{
      const resp = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      if(!resp.ok){
        const err = await resp.json().catch(()=>({error:'unknown'}));
        out.textContent = 'Error: '+(err.error||resp.statusText);
        out.style.opacity=1;
        return;
      }
      const data = await resp.json();
      out.textContent = `Prediction: ${data.prediction}`;
      out.style.opacity=1;
    }catch(e){
      out.textContent='Network error: could not reach server';
      out.style.opacity=1;
    }
  });
});