"use client"

import { useEffect, useState, use } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, ChevronRight, Download, Activity, FileText } from "lucide-react"

export default function ResultsPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter()
  const unwrappedParams = use(params)
  const [result, setResult] = useState<any>(null)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.7)

  useEffect(() => {
    // In a real app, this would be a fetch to /api/screenings/:id
    const stored = sessionStorage.getItem(`scan_${unwrappedParams.id}`)
    if (stored) {
      setResult(JSON.parse(stored))
    } else {
      // Fallback for demo
      setResult({
        prediction: "melanoma",
        confidence: 0.82,
        uncertainty: "moderate",
        top_predictions: [
          { class: "melanoma", probability: 0.82, name: "Melanoma" },
          { class: "melanocytic_nevus", probability: 0.09, name: "Melanocytic nevus" },
          { class: "basal_cell_carcinoma", probability: 0.05, name: "Basal cell carcinoma" },
          { class: "other", probability: 0.04, name: "Other" }
        ],
        preview: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=800", // Generic skin placeholder
        heatmap_url: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=800", // Would be actual heatmap
      })
    }
  }, [unwrappedParams.id])

  if (!result) {
    return <div className="container py-24 text-center">Loading...</div>
  }

  const confidencePercentage = Math.round(result.confidence * 100)

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
        <div>
          <div className="inline-flex items-center rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary mb-4">
            Preliminary AI Assessment
          </div>
          <h1 className="text-3xl font-bold">AI Screening Result</h1>
          <p className="text-muted-foreground mt-2">
            These probabilities represent model outputs, not confirmed diagnoses.
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" /> Download Report
          </Button>
          <Button asChild>
            <Link href={`/results/${unwrappedParams.id}/questionnaire`}>
              Add Symptoms <ChevronRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Result & Differential */}
        <div className="lg:col-span-1 space-y-6">
          <Card className="glass shadow-lg shadow-black/20 border-primary/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Possible Condition</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400 mb-4 capitalize">
                {result.top_predictions[0].name}
              </div>
              
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground font-medium">Confidence</span>
                  <span className="font-bold text-white">{confidencePercentage}%</span>
                </div>
                <div className="h-2.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/10">
                  <div 
                    className="h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-cyan-500 to-blue-600 shadow-[0_0_10px_rgba(0,212,255,0.5)]"
                    style={{ width: `${confidencePercentage}%` }}
                  />
                </div>
              </div>
              
              <div className="mt-5 pt-4 border-t border-white/10 flex items-start gap-3 bg-white/5 p-3 rounded-lg">
                <Activity className="h-5 w-5 text-yellow-500 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-white">AI Certainty: <span className="text-yellow-400 capitalize">{result.uncertainty}</span></p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Possible {result.top_predictions[0].name} — preliminary AI assessment.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass">
            <CardHeader>
              <CardTitle>AI Differential</CardTitle>
              <CardDescription>Other possibilities considered</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result.top_predictions.map((pred: any, idx: number) => (
                <div key={idx} className="space-y-1.5">
                  <div className="flex justify-between text-sm">
                    <span className="text-white/90 font-medium">{pred.name}</span>
                    <span className="text-muted-foreground">{Math.round(pred.probability * 100)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${idx === 0 ? 'bg-gradient-to-r from-cyan-500 to-blue-500' : 'bg-white/20'}`}
                      style={{ width: `${Math.round(pred.probability * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-destructive/10 border-destructive/30 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="text-destructive flex items-center gap-2">
                <AlertCircle className="h-5 w-5" /> Suggested Next Step
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm font-medium text-white mb-2">MODERATE URGENCY</p>
              <p className="text-sm text-destructive/80">
                Consider professional evaluation soon. This is a screening recommendation and not a medical diagnosis.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Explainable AI */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="h-full flex flex-col glass">
            <CardHeader className="pb-4">
              <CardTitle>Explainable AI</CardTitle>
              <CardDescription>Visualizing areas that contributed to the prediction.</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <div className="relative rounded-xl overflow-hidden bg-black/60 border border-white/10 aspect-square md:aspect-video flex-1 flex items-center justify-center">
                
                {/* Floating controls */}
                <div className="absolute top-4 right-4 z-20 flex items-center gap-1 bg-black/60 backdrop-blur-md p-1 rounded-full border border-white/10 shadow-lg">
                  <Button 
                    variant={showHeatmap ? "ghost" : "secondary"} 
                    size="sm" 
                    onClick={() => setShowHeatmap(false)}
                    className="text-xs h-8 rounded-full px-4 transition-all"
                  >
                    Original
                  </Button>
                  <Button 
                    variant={showHeatmap ? "secondary" : "ghost"} 
                    size="sm" 
                    onClick={() => setShowHeatmap(true)}
                    className="text-xs h-8 rounded-full px-4 transition-all bg-primary/20 text-primary hover:bg-primary/30"
                  >
                    Attention Map
                  </Button>
                </div>
                {/* Original Image */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={result.preview} 
                  alt="Original Skin Image" 
                  className="absolute inset-0 w-full h-full object-contain" 
                />
                
                {/* Heatmap Overlay (Mocked using CSS gradient over the image) */}
                {showHeatmap && (
                  <div 
                    className="absolute inset-0 z-10 mix-blend-screen pointer-events-none"
                    style={{ 
                      opacity: heatmapOpacity,
                      background: 'radial-gradient(circle at 50% 50%, rgba(255,0,0,0.8) 0%, rgba(255,255,0,0.4) 20%, rgba(0,0,255,0.1) 50%, transparent 70%)',
                    }}
                  />
                )}
              </div>
              
              {showHeatmap && (
                <div className="mt-6 flex items-center gap-4">
                  <span className="text-xs text-muted-foreground shrink-0">Overlay Opacity</span>
                  <input 
                    type="range" 
                    min="0" max="1" step="0.1" 
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                    className="w-full accent-primary h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              )}

              <div className="mt-6 bg-white/5 p-4 rounded-lg border border-white/10">
                <p className="text-sm text-muted-foreground flex gap-3">
                  <FileText className="h-5 w-5 text-primary shrink-0" />
                  <span>
                    Highlighted regions indicate areas that contributed strongly to the model&apos;s prediction. 
                    This visualization is intended for interpretability and does not represent a medical diagnosis.
                  </span>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
