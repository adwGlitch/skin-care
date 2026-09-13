"use client"

import { useEffect, useState, use } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Printer, ShieldCheck, FileText } from "lucide-react"

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const unwrappedParams = use(params)
  const [result, setResult] = useState<any>(null)
  
  useEffect(() => {
    const stored = sessionStorage.getItem(`scan_${unwrappedParams.id}`)
    if (stored) {
      setResult(JSON.parse(stored))
    }
  }, [unwrappedParams.id])

  if (!result) {
    return <div className="container py-24 text-center">Loading Report...</div>
  }

  const printReport = () => {
    window.print()
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <div className="flex justify-between items-center mb-8 print:hidden">
        <Button variant="ghost" asChild className="-ml-4 text-muted-foreground">
          <Link href={`/results/${unwrappedParams.id}`}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Results
          </Link>
        </Button>
        <Button onClick={printReport}>
          <Printer className="mr-2 h-4 w-4" /> Print PDF
        </Button>
      </div>

      <div className="bg-white text-black p-8 md:p-12 rounded-xl print:p-0 print:shadow-none print:m-0" id="report-content">
        {/* Report Header */}
        <div className="border-b-2 border-gray-200 pb-8 mb-8 flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 mb-2">DERMAAI</h1>
            <p className="text-gray-500 font-medium tracking-widest uppercase text-sm">AI-Assisted Skin Screening Report</p>
          </div>
          <div className="text-right text-sm text-gray-500 space-y-1">
            <p><strong>Date:</strong> {new Date().toLocaleDateString()}</p>
            <p><strong>Scan ID:</strong> {unwrappedParams.id.toUpperCase()}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div>
            <h3 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">Image Information</h3>
            <div className="bg-gray-100 rounded-lg aspect-video mb-4 overflow-hidden flex items-center justify-center relative">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={result.preview} alt="Scan preview" className="w-full h-full object-contain" />
            </div>
            <p className="text-sm text-gray-600">
              <strong>Quality Score:</strong> {result.image_quality?.score || 92}/100 ({result.image_quality?.status || 'Good'})
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">AI Preliminary Assessment</h3>
            <div className="bg-blue-50 border border-blue-100 p-4 rounded-lg mb-6">
              <p className="text-sm text-blue-600 font-medium uppercase mb-1">Top Possibility</p>
              <p className="text-2xl font-bold text-blue-900 capitalize mb-2">{result.top_predictions[0].name}</p>
              <p className="text-sm text-gray-600">Model Confidence: <strong>{Math.round(result.confidence * 100)}%</strong></p>
            </div>
            
            <h4 className="text-sm font-bold text-gray-900 mb-2">AI Differential</h4>
            <div className="space-y-2">
              {result.top_predictions.slice(1).map((p: any, i: number) => (
                <div key={i} className="flex justify-between text-sm items-center">
                  <span className="text-gray-700">{p.name}</span>
                  <span className="text-gray-500 bg-gray-100 px-2 py-0.5 rounded text-xs font-mono">{Math.round(p.probability * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Symptoms Section */}
        {result.symptoms && (
          <div className="mb-8">
            <h3 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">Reported Context & Symptoms</h3>
            <div className="grid grid-cols-2 gap-4 text-sm bg-gray-50 p-4 rounded-lg">
              <div>
                <p className="text-gray-500 mb-1">Duration</p>
                <p className="font-medium text-gray-900">{result.symptoms.duration || 'Not specified'}</p>
              </div>
              <div>
                <p className="text-gray-500 mb-1">Recent Changes</p>
                <p className="font-medium text-gray-900">{result.symptoms.changed || 'Not specified'}</p>
              </div>
              <div className="col-span-2">
                <p className="text-gray-500 mb-1">Symptoms</p>
                <p className="font-medium text-gray-900">
                  {result.symptoms.symptoms?.length > 0 ? result.symptoms.symptoms.join(', ') : 'None reported'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Triage */}
        <div className="mb-8">
          <h3 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">Suggested Next Step</h3>
          <div className={`p-4 rounded-lg border-l-4 ${
            result.triage === 'HIGH' ? 'bg-red-50 border-red-500 text-red-900' :
            result.triage === 'MODERATE' ? 'bg-orange-50 border-orange-500 text-orange-900' :
            'bg-yellow-50 border-yellow-500 text-yellow-900'
          }`}>
            <p className="font-bold mb-1 uppercase tracking-wide text-sm">{result.triage || 'MODERATE'} URGENCY</p>
            <p className="text-sm">
              {result.triage === 'HIGH' ? 'Prompt professional evaluation is recommended.' : 
               'Consider professional evaluation soon.'}
            </p>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-12 pt-8 border-t-2 border-gray-200">
          <div className="flex items-start gap-4">
            <ShieldCheck className="h-6 w-6 text-gray-400 shrink-0" />
            <p className="text-xs text-gray-500 leading-relaxed font-medium">
              <strong>IMPORTANT DISCLAIMER:</strong> This report contains a preliminary AI-assisted screening assessment. 
              It is not a confirmed medical diagnosis and should not be used as a substitute for evaluation by a qualified healthcare professional. 
              The AI model provides probabilities based on visual similarity to its training data, but it cannot replace clinical judgment, 
              biopsies, or other diagnostic tests. If you notice any changing, bleeding, or concerning skin lesions, seek medical attention regardless of this report.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
