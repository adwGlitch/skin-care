"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { BadgeCheck, XCircle, HelpCircle, FileText, Filter } from "lucide-react"

const cases = [
  {
    id: "CASE-092",
    date: "12 Sep 2026",
    prediction: "Melanoma",
    confidence: "82%",
    status: "pending",
    img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=300"
  },
  {
    id: "CASE-091",
    date: "11 Sep 2026",
    prediction: "Basal cell carcinoma",
    confidence: "65%",
    status: "pending",
    img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=300"
  },
  {
    id: "CASE-090",
    date: "10 Sep 2026",
    prediction: "Melanocytic nevus",
    confidence: "95%",
    status: "reviewed",
    img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=300"
  }
]

export default function ProfessionalDashboard() {
  const [activeTab, setActiveTab] = useState<"pending" | "reviewed">("pending")
  const [reviewedCases, setReviewedCases] = useState<string[]>([])

  const filteredCases = cases.filter(c => 
    activeTab === "pending" 
      ? c.status === "pending" && !reviewedCases.includes(c.id)
      : c.status === "reviewed" || reviewedCases.includes(c.id)
  )

  const handleReview = (id: string) => {
    setReviewedCases([...reviewedCases, id])
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-6xl">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
        <div>
          <div className="inline-flex items-center rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-400 mb-2">
            Clinical Interface
          </div>
          <h1 className="text-3xl font-bold">Professional Review Dashboard</h1>
          <p className="text-muted-foreground mt-1">Human-in-the-loop verification for AI model improvement.</p>
        </div>
      </div>

      <div className="flex gap-2 border-b border-white/10 mb-8 pb-2 overflow-x-auto">
        <Button 
          variant={activeTab === "pending" ? "default" : "ghost"} 
          onClick={() => setActiveTab("pending")}
          className="rounded-full"
        >
          Pending Review ({cases.filter(c => c.status === "pending" && !reviewedCases.includes(c.id)).length})
        </Button>
        <Button 
          variant={activeTab === "reviewed" ? "default" : "ghost"} 
          onClick={() => setActiveTab("reviewed")}
          className="rounded-full"
        >
          Reviewed Cases
        </Button>
        <div className="ml-auto flex items-center">
          <Button variant="outline" size="sm" className="hidden md:flex">
            <Filter className="mr-2 h-4 w-4" /> Filter
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {filteredCases.map(c => (
          <Card key={c.id} className="overflow-hidden glass flex flex-col md:flex-row">
            <div className="md:w-64 bg-black/60 shrink-0 relative aspect-video md:aspect-auto border-r border-white/5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={c.img} alt="Clinical case" className="absolute inset-0 w-full h-full object-cover" />
            </div>
            
            <div className="flex-1 p-6 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-xl font-bold">{c.id}</h3>
                    <p className="text-sm text-muted-foreground">{c.date}</p>
                  </div>
                  <div className="bg-primary/20 text-primary px-3 py-1 rounded-full text-sm font-medium">
                    AI: {c.prediction} ({c.confidence})
                  </div>
                </div>
                
                <p className="text-sm text-white/80 mb-6 max-w-2xl">
                  <strong>Patient reported:</strong> Changed recently (Yes), Duration (1-6 months), Symptoms (Itching). 
                  AI differential included Melanocytic nevus (9%) and Basal cell carcinoma (5%).
                </p>
              </div>

              {activeTab === "pending" ? (
                <div className="bg-black/30 p-5 rounded-xl border border-white/5">
                  <p className="text-sm font-medium mb-4 text-white/80">Model Verification (Human-in-the-loop)</p>
                  <div className="flex flex-wrap gap-3">
                    <Button 
                      variant="outline" 
                      className="border-emerald-500/20 hover:bg-emerald-500/10 text-emerald-400 bg-emerald-500/5 transition-colors"
                      onClick={() => handleReview(c.id)}
                    >
                      <BadgeCheck className="mr-2 h-4 w-4" /> AI Agrees
                    </Button>
                    <Button 
                      variant="outline" 
                      className="border-rose-500/20 hover:bg-rose-500/10 text-rose-400 bg-rose-500/5 transition-colors"
                      onClick={() => handleReview(c.id)}
                    >
                      <XCircle className="mr-2 h-4 w-4" /> AI Incorrect
                    </Button>
                    <Button 
                      variant="outline" 
                      className="border-amber-500/20 hover:bg-amber-500/10 text-amber-400 bg-amber-500/5 transition-colors"
                      onClick={() => handleReview(c.id)}
                    >
                      <HelpCircle className="mr-2 h-4 w-4" /> Uncertain
                    </Button>
                    <Button variant="ghost" className="ml-auto text-muted-foreground hover:text-white" size="sm">
                      <FileText className="mr-2 h-4 w-4" /> View Full Report
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/20 flex items-center justify-between">
                  <div className="flex items-center text-green-400">
                    <BadgeCheck className="mr-2 h-5 w-5" />
                    <span className="font-medium">Verified by Dr. Smith</span>
                  </div>
                  <Button variant="ghost" size="sm">Edit Assessment</Button>
                </div>
              )}
            </div>
          </Card>
        ))}

        {filteredCases.length === 0 && (
          <div className="text-center py-24 text-muted-foreground bg-white/5 rounded-xl border border-white/10">
            No {activeTab} cases found.
          </div>
        )}
      </div>
    </div>
  )
}
