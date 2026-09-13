"use client"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { UploadCloud, Camera, Image as ImageIcon, AlertCircle, CheckCircle2, Loader2 } from "lucide-react"

type ScreeningStage = "upload" | "quality_check" | "processing" | "error"

export default function ScreeningPage() {
  const router = useRouter()
  const [stage, setStage] = useState<ScreeningStage>("upload")
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [qualityScore, setQualityScore] = useState<number | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Pipeline animation states
  const [pipelineSteps, setPipelineSteps] = useState([
    { name: "Image Received", status: "pending" },
    { name: "Rescaling to 224 × 224", status: "pending" },
    { name: "Preprocessing", status: "pending" },
    { name: "Segmentation", status: "pending" },
    { name: "Feature Extraction", status: "pending" },
    { name: "Attention Analysis", status: "pending" },
    { name: "Classification", status: "pending" },
  ])
  
  const currentStepIndex = pipelineSteps.findIndex(s => s.status === "processing" || s.status === "pending")

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      // Validate file type
      if (!selectedFile.type.startsWith('image/')) {
        setErrorMsg("Please upload a valid image file (JPG, PNG).")
        setStage("error")
        return
      }
      
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setStage("quality_check")
      simulateQualityCheck()
    }
  }

  const simulateQualityCheck = () => {
    setQualityScore(null)
    setTimeout(() => {
      // Mock: 90% chance of good quality, 10% chance of failing
      const isGood = Math.random() > 0.1
      if (isGood) {
        setQualityScore(85 + Math.floor(Math.random() * 10))
      } else {
        setErrorMsg("Image quality is insufficient for reliable analysis. The image appears blurry or lighting is poor.")
        setStage("error")
      }
    }, 1500)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setStage("processing")
    
    // Simulate pipeline steps animation
    let step = 0
    const interval = setInterval(() => {
      setPipelineSteps(prev => prev.map((p, i) => {
        if (i < step) return { ...p, status: "completed" }
        if (i === step) return { ...p, status: "processing" }
        return p
      }))
      step++
      if (step >= pipelineSteps.length) {
        clearInterval(interval)
      }
    }, 500)

    try {
      const formData = new FormData()
      formData.append("image", file)
      
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })
      
      if (!res.ok) throw new Error("Analysis failed")
      
      const data = await res.json()
      
      // Complete final step
      setPipelineSteps(prev => prev.map(p => ({ ...p, status: "completed" })))
      
      // Store result in sessionStorage (since no real DB for prototype)
      sessionStorage.setItem(`scan_${data.scan_id}`, JSON.stringify({
        ...data,
        preview,
      }))
      
      setTimeout(() => {
        router.push(`/results/${data.scan_id}`)
      }, 500)
      
    } catch (err) {
      clearInterval(interval)
      setErrorMsg("An error occurred during analysis. Please try again.")
      setStage("error")
    }
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-3xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">AI Skin Screening</h1>
        <p className="text-muted-foreground">Upload a clear image of the affected skin area for preliminary AI-assisted analysis.</p>
      </div>

      <Card className="overflow-hidden bg-black/20 border-white/10 backdrop-blur-md">
        <CardContent className="p-0">
          {stage === "upload" && (
            <div 
              className="p-12 border-2 border-dashed border-white/20 m-6 rounded-xl text-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer bg-white/5 relative group overflow-hidden"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="absolute inset-0 bg-gradient-to-t from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
              <UploadCloud className="h-16 w-16 mx-auto mb-4 text-primary/70 group-hover:text-primary transition-colors group-hover:scale-110 duration-300" />
              <h3 className="text-xl font-medium mb-2 text-white">Drag and drop or click to upload</h3>
              <p className="text-sm text-muted-foreground mb-6 font-light">Supported formats: JPG, JPEG, PNG, WebP</p>
              
              <div className="flex justify-center gap-4">
                <Button variant="secondary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}>
                  <ImageIcon className="mr-2 h-4 w-4" /> Browse Files
                </Button>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="image/*"
                onChange={handleFileSelect}
              />
              <p className="mt-8 text-xs text-yellow-500/80 bg-yellow-500/10 inline-block px-3 py-1 rounded-full">
                Do not upload images containing unnecessary personal information (e.g., faces, tattoos) unless relevant.
              </p>
            </div>
          )}

          {stage === "quality_check" && preview && (
            <div className="p-6">
              <div className="relative aspect-square md:aspect-video rounded-lg overflow-hidden bg-black mb-6 flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain" />
              </div>
              
              <div className="flex flex-col items-center justify-center p-6 bg-white/5 rounded-lg border border-white/10 mb-6">
                {!qualityScore ? (
                  <div className="flex flex-col items-center">
                    <Loader2 className="h-8 w-8 text-primary animate-spin mb-4" />
                    <h3 className="text-lg font-medium">Assessing Image Quality...</h3>
                    <p className="text-sm text-muted-foreground">Checking lighting, focus, and framing.</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center w-full">
                    <CheckCircle2 className="h-12 w-12 text-green-500 mb-2" />
                    <h3 className="text-xl font-bold mb-1">Image Quality: Good</h3>
                    <div className="w-full max-w-md bg-white/10 rounded-full h-2 mb-2 mt-4">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: `${qualityScore}%` }}></div>
                    </div>
                    <p className="text-sm text-muted-foreground">Quality Score: {qualityScore}/100</p>
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-4">
                <Button variant="outline" onClick={() => setStage("upload")}>
                  Replace Image
                </Button>
                <Button onClick={handleAnalyze} disabled={!qualityScore} className="bg-primary hover:bg-primary/90">
                  Continue Analysis
                </Button>
              </div>
            </div>
          )}

          {stage === "processing" && preview && (
            <div className="p-6 md:p-12">
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div className="relative rounded-lg overflow-hidden border border-white/10 aspect-square group">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Processing" className="w-full h-full object-cover opacity-50" />
                  
                  {/* Scanning Animation */}
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/30 to-transparent w-full h-[20%] animate-scan mix-blend-screen"></div>
                  <div className="absolute inset-0 border-2 border-primary/40 m-8 rounded-lg border-dashed animate-pulse-glow">
                    <div className="absolute top-0 left-0 w-6 h-6 border-t-4 border-l-4 border-primary -mt-[2px] -ml-[2px] rounded-tl-sm"></div>
                    <div className="absolute top-0 right-0 w-6 h-6 border-t-4 border-r-4 border-primary -mt-[2px] -mr-[2px] rounded-tr-sm"></div>
                    <div className="absolute bottom-0 left-0 w-6 h-6 border-b-4 border-l-4 border-primary -mb-[2px] -ml-[2px] rounded-bl-sm"></div>
                    <div className="absolute bottom-0 right-0 w-6 h-6 border-b-4 border-r-4 border-primary -mb-[2px] -mr-[2px] rounded-br-sm"></div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white mb-6 flex items-center">
                    <Loader2 className="mr-3 h-5 w-5 animate-spin text-primary" /> 
                    AI Processing Pipeline
                  </h3>
                  
                  <div className="space-y-3">
                    {pipelineSteps.map((step, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        {step.status === "completed" && <CheckCircle2 className="h-5 w-5 text-green-500" />}
                        {step.status === "processing" && <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />}
                        {step.status === "pending" && <div className="h-5 w-5 rounded-full border-2 border-white/20" />}
                        
                        <span className={`text-sm font-mono ${step.status === 'completed' ? 'text-white/80' : step.status === 'processing' ? 'text-primary font-bold' : 'text-white/40'}`}>
                          {step.status === "processing" ? "> " : ""}{step.name}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {stage === "error" && (
            <div className="p-12 text-center flex flex-col items-center">
              <AlertCircle className="h-16 w-16 text-destructive mb-4" />
              <h3 className="text-xl font-bold mb-2">Analysis Error</h3>
              <p className="text-muted-foreground mb-8 max-w-md">{errorMsg}</p>
              <Button onClick={() => setStage("upload")}>
                Retake Image
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
