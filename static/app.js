const { useState, useEffect, useRef } = React;

const SAMPLE_PRESETS = [
  { name: 'Melanoma Suspect', age: 62, sex: 'male', localization: 'back', color: '#3d1e08' },
  { name: 'Benign Nevus', age: 28, sex: 'female', localization: 'lower extremity', color: '#8c532b' },
  { name: 'Basal Cell Carcinoma', age: 71, sex: 'male', localization: 'face', color: '#d9777f' },
  { name: 'Actinic Keratosis', age: 66, sex: 'female', localization: 'scalp', color: '#b45309' }
];

const BODY_REGIONS = [
  { id: 'face', label: 'Face', icon: '👤' },
  { id: 'scalp', label: 'Scalp', icon: '💇' },
  { id: 'neck', label: 'Neck', icon: '🧣' },
  { id: 'chest', label: 'Chest', icon: '🫁' },
  { id: 'back', label: 'Back', icon: '🛡️' },
  { id: 'abdomen', label: 'Abdomen', icon: '🩻' },
  { id: 'trunk', label: 'Trunk', icon: '👕' },
  { id: 'upper extremity', label: 'Upper Arm', icon: '💪' },
  { id: 'lower extremity', label: 'Leg / Thigh', icon: '🦵' },
  { id: 'hand', label: 'Hand', icon: '✋' },
  { id: 'foot', label: 'Foot', icon: '🦶' },
  { id: 'ear', label: 'Ear', icon: '👂' },
  { id: 'genital', label: 'Genital', icon: '🔒' },
  { id: 'unknown', label: 'Unknown', icon: '❓' }
];

function App() {
  const [activeTab, setActiveTab] = useState('diagnostics');
  const [healthInfo, setHealthInfo] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [classesInfo, setClassesInfo] = useState([]);
  const [researchData, setResearchData] = useState(null);
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageViewMode, setImageViewMode] = useState('heatmap');
  const [age, setAge] = useState(48);
  const [sex, setSex] = useState('female');
  const [localization, setLocalization] = useState('back');
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const [abcde, setAbcde] = useState({
    asymmetry: false,
    border: false,
    color: false,
    diameter: false,
    evolving: false
  });

  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const benchmarkChartRef = useRef(null);
  const benchmarkChartInstance = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setHealthInfo).catch(console.error);
    fetch('/api/classes').then(r => r.json()).then(d => setClassesInfo(d.classes || [])).catch(console.error);
    fetch('/api/model-info').then(r => r.json()).then(setModelInfo).catch(console.error);
    fetch('/api/research-data').then(r => r.json()).then(setResearchData).catch(console.error);
  }, []);

  useEffect(() => {
    if (predictionResult && chartRef.current) {
      if (chartInstance.current) chartInstance.current.destroy();

      const labels = predictionResult.class_breakdown.map(c => c.name);
      const dataValues = predictionResult.class_breakdown.map(c => c.percentage);
      const bgColors = predictionResult.class_breakdown.map(c => {
        if (c.risk_color === 'rose') return 'rgba(244, 63, 94, 0.9)';
        if (c.risk_color === 'amber') return 'rgba(245, 158, 11, 0.9)';
        return 'rgba(16, 185, 129, 0.9)';
      });

      const ctx = chartRef.current.getContext('2d');
      chartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            data: dataValues,
            backgroundColor: bgColors,
            borderRadius: 6
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
            y: { ticks: { color: '#f1f5f9', font: { weight: '600', size: 11 } }, grid: { display: false } }
          }
        }
      });
    }
  }, [predictionResult, activeTab]);

  useEffect(() => {
    if (activeTab === 'architecture' && benchmarkChartRef.current && modelInfo?.metrics_summary) {
      if (benchmarkChartInstance.current) benchmarkChartInstance.current.destroy();

      const parsePct = str => parseFloat(String(str || '0').replace('%', '')) || 0;
      const labels = modelInfo.metrics_summary.map(m => m.model.split(':')[0]);
      const accData = modelInfo.metrics_summary.map(m => parsePct(m.accuracy));
      const balAccData = modelInfo.metrics_summary.map(m => parsePct(m.balanced_accuracy));
      const melRecallData = modelInfo.metrics_summary.map(m => parsePct(m.melanoma_recall));
      const melSpecData = modelInfo.metrics_summary.map(m => parsePct(m.melanoma_specificity));

      const ctx = benchmarkChartRef.current.getContext('2d');
      benchmarkChartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            { label: 'Overall Acc (%)', data: accData, backgroundColor: 'rgba(56, 189, 248, 0.85)', borderRadius: 4 },
            { label: 'Balanced Acc (%)', data: balAccData, backgroundColor: 'rgba(20, 184, 166, 0.85)', borderRadius: 4 },
            { label: 'Melanoma Recall (%)', data: melRecallData, backgroundColor: 'rgba(244, 63, 94, 0.85)', borderRadius: 4 },
            { label: 'Melanoma Specificity (%)', data: melSpecData, backgroundColor: 'rgba(168, 85, 247, 0.85)', borderRadius: 4 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 11, weight: 'bold' } } }
          },
          scales: {
            y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.06)' } },
            x: { ticks: { color: '#f1f5f9', font: { weight: '600' } }, grid: { display: false } }
          }
        }
      });
    }
  }, [activeTab, modelInfo]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const applyPreset = (p) => {
    setAge(p.age);
    setSex(p.sex);
    setLocalization(p.localization);
  };

  const handleDiagnose = async () => {
    if (!selectedFile) {
      setErrorMsg('Please upload a dermoscopic image before initiating analysis.');
      return;
    }
    setErrorMsg(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('age', age);
      formData.append('sex', sex);
      formData.append('localization', localization);

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await res.json();
      setPredictionResult(data);
      setImageViewMode('heatmap');
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const abcdeCount = Object.values(abcde).filter(Boolean).length;

  return (
    <div className='min-h-screen flex flex-col font-sans'>
      {/* Top Navigation */}
      <header className='border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-50'>
        <div className='max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap justify-between items-center gap-3'>
          <div className='flex items-center space-x-3'>
            <div className='w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-teal-500/20 text-white font-black text-xl'>
              DF
            </div>
            <div>
              <h1 className='text-base font-extrabold text-white tracking-tight flex items-center space-x-2'>
                <span>DermaFusion AI</span>
                <span className='px-2 py-0.5 text-[10px] font-bold bg-teal-950 text-teal-300 rounded border border-teal-800/80 uppercase'>
                  Research v2.1
                </span>
              </h1>
              <p className='text-xs text-slate-400'>Leakage-Controlled Multimodal Skin Lesion Diagnostic System</p>
            </div>
          </div>

          <nav className='flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800'>
            {[
              { id: 'diagnostics', label: '🔬 Live Diagnostic Engine' },
              { id: 'architecture', label: '📊 Research Benchmark Suite' },
              { id: 'atlas', label: '📖 HAM10000 Encyclopedia' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={'px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ' + (activeTab === tab.id ? 'bg-teal-500 text-white shadow-md shadow-teal-500/25' : 'text-slate-400 hover:text-slate-200')}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Body */}
      <main className='flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6'>
        
        {/* TAB 1: DIAGNOSTIC WORKBENCH */}
        {activeTab === 'diagnostics' && (
          <div className='grid grid-cols-1 lg:grid-cols-12 gap-6'>
            
            {/* LEFT COLUMN: Input Form */}
            <div className='lg:col-span-5 space-y-4'>
              
              {/* Presets */}
              <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl'>
                <span className='text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2'>
                  ⚡ Clinical Case Presets
                </span>
                <div className='grid grid-cols-2 gap-2'>
                  {SAMPLE_PRESETS.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => applyPreset(p)}
                      className='text-left p-2.5 rounded-xl border border-slate-800/80 bg-slate-950/60 hover:border-teal-500/50 hover:bg-teal-950/20 transition-all text-xs group'
                    >
                      <div className='font-bold text-slate-200 group-hover:text-teal-300'>{p.name}</div>
                      <div className='text-[10px] text-slate-500'>{p.age}y • {p.sex} • {p.localization}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Step 1: Image Upload */}
              <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <div className='flex justify-between items-center'>
                  <h2 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                    <span className='w-2 h-2 rounded-full bg-teal-400'></span>
                    <span>1. Dermoscopic Image Ingestion</span>
                  </h2>
                  <span className='text-[10px] text-slate-400'>DullRazor Filter Enabled</span>
                </div>

                <div
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  className={'border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[160px] ' + (imagePreview ? 'border-teal-500/50 bg-teal-950/10' : 'border-slate-800 bg-slate-950 hover:border-slate-700')}
                >
                  <input ref={fileInputRef} type='file' accept='image/*' onChange={handleFileChange} className='hidden' />
                  {imagePreview ? (
                    <div className='relative group'>
                      <img src={imagePreview} alt='Preview' className='h-32 w-32 object-cover rounded-lg border border-slate-700 shadow-md' />
                      <div className='absolute inset-0 bg-black/60 rounded-lg opacity-0 group-hover:opacity-100 flex items-center justify-center text-[10px] font-bold text-white transition-opacity'>
                        Change Image
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className='text-2xl mb-1 text-teal-400'>📸</div>
                      <span className='text-xs font-bold text-slate-300'>Upload Dermoscopic Photograph</span>
                      <span className='text-[10px] text-slate-500 mt-0.5'>Supports JPEG, PNG • HAM10000 format</span>
                    </>
                  )}
                </div>
              </div>

              {/* Step 2: Metadata */}
              <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <h2 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                  <span className='w-2 h-2 rounded-full bg-cyan-400'></span>
                  <span>2. Patient Metadata (19 Gated Features)</span>
                </h2>

                <div>
                  <div className='flex justify-between text-xs mb-1 font-semibold'>
                    <span className='text-slate-300'>Patient Age</span>
                    <span className='text-teal-400 bg-teal-950/60 px-2 py-0.5 rounded border border-teal-800/50'>{age} years</span>
                  </div>
                  <input type='range' min='1' max='95' value={age} onChange={(e) => setAge(parseInt(e.target.value))} className='w-full accent-teal-500' />
                </div>

                <div>
                  <label className='block text-xs font-semibold text-slate-300 mb-1.5'>Patient Sex</label>
                  <div className='grid grid-cols-3 gap-2'>
                    {['female', 'male', 'unknown'].map((s) => (
                      <button
                        key={s}
                        type='button'
                        onClick={() => setSex(s)}
                        className={'py-2 rounded-lg text-xs font-bold uppercase border transition-all ' + (sex === s ? 'bg-teal-500/20 border-teal-500 text-teal-300' : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700')}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <div className='flex justify-between items-center mb-1.5'>
                    <label className='text-xs font-semibold text-slate-300'>Anatomical Lesion Site</label>
                    <span className='text-[10px] text-cyan-400 font-bold uppercase bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/50'>
                      Selected: {localization}
                    </span>
                  </div>
                  
                  <div className='grid grid-cols-2 sm:grid-cols-3 gap-1.5 max-h-44 overflow-y-auto p-1 bg-slate-950 rounded-xl border border-slate-800'>
                    {BODY_REGIONS.map((r) => (
                      <button
                        key={r.id}
                        type='button'
                        onClick={() => setLocalization(r.id)}
                        className={'p-2 rounded-lg text-left text-xs flex items-center space-x-2 border transition-all ' + (localization === r.id ? 'bg-teal-500/20 border-teal-500 text-teal-300 font-bold' : 'bg-slate-900/60 border-slate-800/80 text-slate-400 hover:text-slate-200')}
                      >
                        <span>{r.icon}</span>
                        <span className='truncate'>{r.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className='pt-2 border-t border-slate-800 space-y-2'>
                  <div className='flex justify-between items-center'>
                    <span className='text-xs font-bold text-slate-300 uppercase'>ABCDE Melanoma Checklist</span>
                    <span className={'text-[10px] font-bold px-2 py-0.5 rounded ' + (abcdeCount >= 2 ? 'bg-rose-950 text-rose-300 border border-rose-700' : 'bg-slate-950 text-slate-400')}>
                      {abcdeCount} criteria flagged
                    </span>
                  </div>
                  <div className='grid grid-cols-2 gap-1.5 text-[11px] text-slate-400'>
                    {[
                      { key: 'asymmetry', label: 'A: Asymmetry' },
                      { key: 'border', label: 'B: Border Irregular' },
                      { key: 'color', label: 'C: Color Variation' },
                      { key: 'diameter', label: 'D: Diameter >6mm' },
                      { key: 'evolving', label: 'E: Evolving Mole' }
                    ].map(item => (
                      <label key={item.key} className='flex items-center space-x-2 p-1.5 bg-slate-950 rounded-lg border border-slate-800/80 cursor-pointer'>
                        <input
                          type='checkbox'
                          checked={abcde[item.key]}
                          onChange={e => setAbcde({ ...abcde, [item.key]: e.target.checked })}
                          className='accent-teal-500 rounded'
                        />
                        <span className={abcde[item.key] ? 'text-teal-300 font-semibold' : ''}>{item.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {errorMsg && <p className='text-xs text-rose-400 bg-rose-950/50 p-2.5 rounded-lg border border-rose-800'>{errorMsg}</p>}

                <button
                  onClick={handleDiagnose}
                  disabled={loading}
                  className='w-full py-3.5 bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-600 hover:from-teal-600 hover:to-blue-700 text-white font-bold text-xs rounded-xl uppercase tracking-wider shadow-lg shadow-teal-500/25 transition-all flex items-center justify-center space-x-2'
                >
                  {loading ? (
                    <span>Running Multimodal Inference...</span>
                  ) : (
                    <>
                      <span>Analyze Skin Lesion</span>
                      <span>⚡</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* RIGHT COLUMN: Output Dashboard */}
            <div className='lg:col-span-7'>
              {predictionResult ? (
                <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl space-y-6'>
                  
                  <div className='flex justify-between items-start border-b border-slate-800 pb-5'>
                    <div>
                      <span className='text-[10px] font-bold text-slate-400 uppercase tracking-wider'>Primary Prediction</span>
                      <h3 className='text-2xl sm:text-3xl font-black text-white mt-1'>{predictionResult.primary_diagnosis.name}</h3>
                      <p className='text-xs text-slate-400 mt-0.5'>{predictionResult.primary_diagnosis.category}</p>
                    </div>
                    <div className='text-right'>
                      <span className={'px-3.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ' + (predictionResult.primary_diagnosis.risk_color === 'rose' ? 'bg-rose-950/80 border-rose-600 text-rose-300' : predictionResult.primary_diagnosis.risk_color === 'amber' ? 'bg-amber-950/80 border-amber-600 text-amber-300' : 'bg-emerald-950/80 border-emerald-600 text-emerald-300')}>
                        {predictionResult.primary_diagnosis.risk_level}
                      </span>
                      <div className='text-3xl font-black text-teal-400 mt-1'>{predictionResult.primary_diagnosis.confidence}%</div>
                      <span className='text-[10px] text-slate-500'>Confidence</span>
                    </div>
                  </div>

                  <div className='p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3'>
                    <div className='flex items-center justify-between'>
                      <div className='flex items-center space-x-2'>
                        <span className='w-2 h-2 rounded-full bg-rose-500 animate-pulse'></span>
                        <h4 className='text-xs font-bold text-slate-200 uppercase tracking-wider'>Explainable AI (Grad-CAM Spatial Heatmap)</h4>
                      </div>
                      <div className='flex space-x-1 bg-slate-900 p-1 rounded-lg border border-slate-800'>
                        <button
                          onClick={() => setImageViewMode('heatmap')}
                          className={'px-2.5 py-1 text-[11px] font-bold rounded ' + (imageViewMode === 'heatmap' ? 'bg-teal-500 text-white' : 'text-slate-400 hover:text-slate-200')}
                        >
                          Heatmap Overlay
                        </button>
                        <button
                          onClick={() => setImageViewMode('original')}
                          className={'px-2.5 py-1 text-[11px] font-bold rounded ' + (imageViewMode === 'original' ? 'bg-teal-500 text-white' : 'text-slate-400 hover:text-slate-200')}
                        >
                          DullRazor Cleaned
                        </button>
                      </div>
                    </div>
                    
                    <div className='flex items-center justify-center p-3 bg-slate-900/60 rounded-xl border border-slate-800/80'>
                      <img
                        src={imageViewMode === 'heatmap' && predictionResult.explainability?.gradcam_overlay ? predictionResult.explainability.gradcam_overlay : imagePreview}
                        alt='Dermoscopic Inspection'
                        className='w-52 h-52 object-cover rounded-xl border border-slate-700 shadow-lg'
                      />
                    </div>
                    <p className='text-[11px] text-center text-slate-400'>
                      {imageViewMode === 'heatmap' 
                        ? '🔥 Red/Yellow contours highlight CNN gradient attention on tumor borders and pigment asymmetry.'
                        : '🧹 Preprocessed dermoscopic image with DullRazor morphological hair/artifact removal.'}
                    </p>
                  </div>

                  <div className='grid grid-cols-1 md:grid-cols-2 gap-4 text-xs'>
                    <div className='p-4 rounded-xl bg-slate-950 border border-slate-800'>
                      <span className='font-bold text-slate-400 uppercase tracking-wider'>Condition Summary</span>
                      <p className='mt-1 text-slate-300 leading-relaxed'>{predictionResult.primary_diagnosis.description}</p>
                    </div>
                    <div className='p-4 rounded-xl bg-slate-950 border border-slate-800'>
                      <span className='font-bold text-slate-400 uppercase tracking-wider'>Recommended Urgency</span>
                      <p className='mt-1 text-slate-300 leading-relaxed'>{predictionResult.primary_diagnosis.urgency}</p>
                    </div>
                  </div>

                  <div className='space-y-2'>
                    <h4 className='text-xs font-bold text-slate-300 uppercase tracking-wider'>Differential Probabilities (7 Diagnostic Classes)</h4>
                    <div className='h-52 bg-slate-950/60 p-3 rounded-xl border border-slate-800'>
                      <canvas ref={chartRef}></canvas>
                    </div>
                  </div>

                  <div className='pt-3 border-t border-slate-800 flex justify-between items-center text-xs text-slate-400'>
                    <span>Engine: <strong className='text-teal-300'>{predictionResult.model_engine}</strong></span>
                    <button onClick={() => window.print()} className='px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl font-bold transition-all no-print flex items-center space-x-1.5'>
                      <span>🖨️</span>
                      <span>Print Clinical Case Report</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className='h-full min-h-[460px] bg-slate-900/40 border-2 border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center p-8 text-center text-slate-400 text-xs space-y-3'>
                  <div className='text-4xl mb-1'>🔬</div>
                  <p className='text-sm font-bold text-slate-200'>Ready for Clinical Lesion Analysis</p>
                  <p className='max-w-xs text-slate-500 leading-relaxed'>Upload a dermoscopic image and select patient details on the left, then click <strong>Analyze Skin Lesion</strong>.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 2: COMPREHENSIVE RESEARCH SUITE */}
        {activeTab === 'architecture' && (
          <div className='space-y-6'>
            
            {/* Header */}
            <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl flex flex-wrap justify-between items-center gap-4'>
              <div>
                <h2 className='text-xl sm:text-2xl font-black text-white'>Multimodal Skin Lesion Empirical Benchmark Suite</h2>
                <p className='text-xs text-slate-400 mt-1'>Zero-leakage patient-lesion clustered validation on HAM10000 dataset ($N=1,452$ test samples).</p>
              </div>
              <div className='flex items-center space-x-2'>
                <span className='text-xs font-bold text-teal-300 bg-teal-950 px-3 py-1 rounded-full border border-teal-700/60'>
                  ✓ Seed 50 Replicable
                </span>
                <span className='text-xs font-bold text-cyan-300 bg-cyan-950 px-3 py-1 rounded-full border border-cyan-700/60'>
                  ✓ Class-Weighted Focal Loss
                </span>
              </div>
            </div>

            {/* CARD 1: 4-Model Comparative Benchmark Matrix */}
            <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
              <div className='flex justify-between items-center'>
                <h3 className='text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                  <span className='w-2.5 h-2.5 rounded-full bg-cyan-400'></span>
                  <span>1. Controlled 4-Model Experimental Benchmark Matrix</span>
                </h3>
                <span className='text-[11px] text-slate-400'>outputs/four_model_benchmark.csv</span>
              </div>

              <div className='h-64 bg-slate-950 p-3 rounded-xl border border-slate-800'>
                <canvas ref={benchmarkChartRef}></canvas>
              </div>

              <div className='overflow-x-auto'>
                <table className='w-full text-left text-xs border-collapse text-slate-300'>
                  <thead>
                    <tr className='bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800 text-[11px]'>
                      <th className='p-3 font-bold'>Model Architecture</th>
                      <th className='p-3 font-bold'>Accuracy</th>
                      <th className='p-3 font-bold'>Balanced Acc</th>
                      <th className='p-3 font-bold'>Macro F1</th>
                      <th className='p-3 font-bold text-cyan-400'>Mel. Precision</th>
                      <th className='p-3 font-bold text-rose-400'>Mel. Recall</th>
                      <th className='p-3 font-bold'>Mel. F1</th>
                      <th className='p-3 font-bold text-purple-400'>Mel. Specificity</th>
                    </tr>
                  </thead>
                  <tbody className='divide-y divide-slate-800'>
                    {(researchData?.four_model_benchmark || modelInfo?.metrics_summary || []).map((m, i) => (
                      <tr key={i} className={String(m['Model'] || m['model']).includes('Proposed') || String(m['Model'] || m['model']).includes('M4') ? 'bg-teal-950/40 text-teal-200 font-bold' : ''}>
                        <td className='p-3'>{m['Model'] || m['model']}</td>
                        <td className='p-3 font-semibold'>{m['Accuracy (%)'] || m['accuracy']}</td>
                        <td className='p-3 font-semibold'>{m['Balanced Acc (%)'] || m['balanced_accuracy']}</td>
                        <td className='p-3'>{m['Macro F1'] || m['macro_f1']}</td>
                        <td className='p-3 text-cyan-400'>{m['Mel. Precision (%)'] || m['melanoma_precision'] || '—'}</td>
                        <td className='p-3 text-rose-400 font-black'>{m['Mel. Recall (%)'] || m['melanoma_recall']}</td>
                        <td className='p-3'>{m['Mel. F1'] || m['melanoma_f1'] || '—'}</td>
                        <td className='p-3 text-purple-400 font-bold'>{m['Mel. Specificity (%)'] || m['melanoma_specificity'] || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* CARD 2 & CARD 3: Statistical Significance & Metadata Ablation */}
            <div className='grid grid-cols-1 lg:grid-cols-2 gap-6'>
              
              {/* Statistical Significance */}
              <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <div className='flex justify-between items-center'>
                  <h3 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                    <span className='w-2 h-2 rounded-full bg-rose-400'></span>
                    <span>2. Statistical Significance (M3 vs. M4)</span>
                  </h3>
                  <span className='text-[10px] text-slate-400'>Exact McNemar & Bootstrap</span>
                </div>

                <div className='space-y-3'>
                  {(researchData?.statistical_significance || []).map((row, i) => (
                    <div key={i} className='p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center text-xs'>
                      <div>
                        <div className='font-bold text-slate-200'>{row['Statistical Evaluation']}</div>
                        <div className='text-[11px] text-slate-400 mt-0.5'>{row['Statistic']}</div>
                      </div>
                      <div className='text-right'>
                        <div className='font-mono font-bold text-teal-300'>{row['Result Metric / Interval']}</div>
                        <span className={'text-[10px] px-2 py-0.5 rounded font-bold uppercase ' + (String(row['Statistically Significant / Excludes Zero']).toLowerCase() === 'true' ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-slate-900 text-slate-400')}>
                          {String(row['Statistically Significant / Excludes Zero']).toLowerCase() === 'true' ? 'CI Excludes 0' : 'p ≥ 0.05 / CI Spans 0'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                <p className='text-[11px] text-slate-400 leading-relaxed bg-slate-950/60 p-3 rounded-xl border border-slate-800/80'>
                  💡 <strong>Clinical Operating Points:</strong> M3 achieves high aggressive sensitivity, while M4 operates as a precision-regularized clinical model that preserves high specificity (83.22%).
                </p>
              </div>

              {/* Metadata Permutation Ablation */}
              <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <div className='flex justify-between items-center'>
                  <h3 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                    <span className='w-2 h-2 rounded-full bg-amber-400'></span>
                    <span>3. Metadata Contribution & Permutation Ablation</span>
                  </h3>
                  <span className='text-[10px] text-slate-400'>M4 Gated Fusion Stream</span>
                </div>

                <div className='overflow-x-auto'>
                  <table className='w-full text-left text-xs border-collapse text-slate-300'>
                    <thead>
                      <tr className='bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800 text-[10px]'>
                        <th className='p-2.5 font-bold'>Condition</th>
                        <th className='p-2.5 font-bold'>Accuracy</th>
                        <th className='p-2.5 font-bold'>Bal. Acc</th>
                        <th className='p-2.5 font-bold text-rose-400'>Mel. Recall</th>
                      </tr>
                    </thead>
                    <tbody className='divide-y divide-slate-800'>
                      {(researchData?.metadata_ablation || []).map((row, i) => (
                        <tr key={i} className={row['Model'].includes('Intact') ? 'bg-teal-950/30 text-teal-200 font-bold' : ''}>
                          <td className='p-2.5'>{row['Model'].split(':')[1] || row['Model']}</td>
                          <td className='p-2.5'>{row['Accuracy (%)']}%</td>
                          <td className='p-2.5'>{row['Balanced Acc (%)']}%</td>
                          <td className='p-2.5 text-rose-400 font-bold'>{row['Mel. Recall (%)']}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className='text-[11px] text-slate-400 leading-relaxed bg-slate-950/60 p-3 rounded-xl border border-slate-800/80'>
                  🔬 <strong>Proof of Clinical Correspondence:</strong> Shuffling patient metadata across mismatched lesions collapses overall accuracy by <strong>-8.75%</strong> (67.77% → 59.02%), proving the network learns true biological age/site relationships.
                </p>
              </div>
            </div>

            {/* CARD 4 & CARD 5: DullRazor Sensitivity & 7-Class Table */}
            <div className='grid grid-cols-1 lg:grid-cols-12 gap-6'>
              
              {/* DullRazor Sensitivity Analysis */}
              <div className='lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <div className='flex justify-between items-center'>
                  <h3 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                    <span className='w-2 h-2 rounded-full bg-emerald-400'></span>
                    <span>4. DullRazor Sensitivity Analysis</span>
                  </h3>
                  <span className='text-[10px] text-slate-400'>M2 Image Model</span>
                </div>

                <div className='overflow-x-auto'>
                  <table className='w-full text-left text-xs border-collapse text-slate-300'>
                    <thead>
                      <tr className='bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800 text-[10px]'>
                        <th className='p-2.5 font-bold'>Preprocessing Stream</th>
                        <th className='p-2.5 font-bold'>Accuracy</th>
                        <th className='p-2.5 font-bold text-rose-400'>Mel. Recall</th>
                      </tr>
                    </thead>
                    <tbody className='divide-y divide-slate-800'>
                      {(researchData?.dullrazor_sensitivity || []).map((row, i) => (
                        <tr key={i} className={row['Model'].includes('Active') ? 'bg-teal-950/30 text-teal-200 font-bold' : ''}>
                          <td className='p-2.5'>{row['Model'].split(':')[1] || row['Model']}</td>
                          <td className='p-2.5'>{row['Accuracy (%)']}%</td>
                          <td className='p-2.5 text-rose-400 font-bold'>{row['Mel. Recall (%)']}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className='p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-400'>
                  ⚠️ Removing DullRazor causes Melanoma recall to collapse from <strong>59.02%</strong> down to <strong>27.87%</strong> (-31.15%), proving morphological inpainting is a critical prerequisite.
                </div>
              </div>

              {/* 7-Class Granular Breakdown */}
              <div className='lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
                <div className='flex justify-between items-center'>
                  <h3 className='text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                    <span className='w-2 h-2 rounded-full bg-indigo-400'></span>
                    <span>5. 7-Class Granular Diagnostic Breakdown (M4 Gated Fusion)</span>
                  </h3>
                  <span className='text-[10px] text-slate-400'>Test Set ($N=1,452$)</span>
                </div>

                <div className='overflow-x-auto'>
                  <table className='w-full text-left text-xs border-collapse text-slate-300'>
                    <thead>
                      <tr className='bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800 text-[10px]'>
                        <th className='p-2 font-bold'>Lesion Code</th>
                        <th className='p-2 font-bold'>Precision</th>
                        <th className='p-2 font-bold'>Recall</th>
                        <th className='p-2 font-bold'>F1-Score</th>
                        <th className='p-2 font-bold'>Support ($N$)</th>
                      </tr>
                    </thead>
                    <tbody className='divide-y divide-slate-800'>
                      {(researchData?.per_class_performance || []).map((row, i) => (
                        <tr key={i} className={row['Class'] === 'MEL' ? 'bg-rose-950/30 text-rose-200 font-bold' : ''}>
                          <td className='p-2 font-bold'>{row['Class']}</td>
                          <td className='p-2'>{row['Precision']}</td>
                          <td className='p-2'>{row['Recall']}</td>
                          <td className='p-2'>{row['F1-Score']}</td>
                          <td className='p-2 text-slate-400 font-mono'>{row['Support']}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* CARD 6: Publication Visualizations Viewer */}
            <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4'>
              <h3 className='text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2'>
                <span className='w-2.5 h-2.5 rounded-full bg-teal-400'></span>
                <span>6. Research Publication Visualizations</span>
              </h3>

              <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div className='bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2'>
                  <h4 className='text-xs font-bold text-slate-200 uppercase'>Figure 1: Training Convergence Curves</h4>
                  <img src='/outputs/training_convergence_curves.png' alt='Learning Curves' className='w-full rounded-lg border border-slate-800 shadow-md' />
                  <p className='text-[10px] text-slate-500'>Validation loss and accuracy trajectories during surgical fine-tuning.</p>
                </div>
                <div className='bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2'>
                  <h4 className='text-xs font-bold text-slate-200 uppercase'>Figure 2: Clinical Grad-CAM Spatial Heatmaps</h4>
                  <img src='/outputs/gradcam_clinical_heatmaps.png' alt='Grad-CAM Activation' className='w-full rounded-lg border border-slate-800 shadow-md' />
                  <p className='text-[10px] text-slate-500'>DullRazor preprocessing and genuine Selvaraju et al. Grad-CAM visual overlays.</p>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* TAB 3: ATLAS */}
        {activeTab === 'atlas' && (
          <div className='bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-xl'>
            <h2 className='text-xl font-black text-white'>HAM10000 Skin Lesion Encyclopedia</h2>
            <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
              {classesInfo.map((c) => (
                <div key={c.id} className='p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs'>
                  <div className='flex justify-between items-center'>
                    <h4 className='font-bold text-white text-sm'>{c.name} ({c.id.toUpperCase()})</h4>
                    <span className={'px-2 py-0.5 rounded text-[10px] font-bold uppercase ' + (c.risk_color === 'rose' ? 'text-rose-400 bg-rose-950 border border-rose-800' : c.risk_color === 'amber' ? 'text-amber-400 bg-amber-950 border border-amber-800' : 'text-emerald-400 bg-emerald-950 border border-emerald-800')}>
                      {c.risk_level}
                    </span>
                  </div>
                  <p className='text-slate-300 leading-relaxed'>{c.description}</p>
                  <p className='text-slate-500 text-[11px]'>Common Sites: {c.common_sites.join(', ')}</p>
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      <footer className='border-t border-slate-800 bg-slate-950 py-6 text-center text-xs text-slate-500 no-print'>
        <p>DermaFusion AI • Research & Clinical Demonstration System for HAM10000 Multimodal Skin Lesion Classification.</p>
        <p className='mt-1 text-slate-600'>Disclaimer: For investigational and academic use only. Not a standalone diagnostic substitute for certified clinical histopathology.</p>
      </footer>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
