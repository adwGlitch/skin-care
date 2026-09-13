"use client"

import { useState, use } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronRight, ArrowLeft } from "lucide-react"
import Link from "next/link"

const questions = [
  {
    id: "duration",
    question: "How long has the skin change been present?",
    options: ["Less than 1 week", "1–4 weeks", "1–6 months", "More than 6 months", "Not sure"]
  },
  {
    id: "changed",
    question: "Has it changed recently? (size, shape, color)",
    options: ["Yes", "No", "Not sure"]
  },
  {
    id: "symptoms",
    question: "Are there any of the following symptoms?",
    options: ["Itching", "Pain", "Bleeding", "Scaling", "Swelling", "None"],
    multi: true
  },
  {
    id: "recurring",
    question: "Is this a recurring condition?",
    options: ["Yes", "No", "Not sure"]
  }
]

export default function QuestionnairePage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter()
  const unwrappedParams = use(params)
  const [answers, setAnswers] = useState<Record<string, any>>({})
  const [currentQ, setCurrentQ] = useState(0)

  const handleSelect = (option: string) => {
    const q = questions[currentQ]
    if (q.multi) {
      const current = answers[q.id] || []
      const updated = current.includes(option)
        ? current.filter((x: string) => x !== option)
        : [...current, option]
      setAnswers({ ...answers, [q.id]: updated })
    } else {
      setAnswers({ ...answers, [q.id]: option })
      if (currentQ < questions.length - 1) {
        setTimeout(() => setCurrentQ(currentQ + 1), 300)
      }
    }
  }

  const handleNext = () => {
    if (currentQ < questions.length - 1) {
      setCurrentQ(currentQ + 1)
    } else {
      // Save and go to report
      const stored = sessionStorage.getItem(`scan_${unwrappedParams.id}`)
      if (stored) {
        const data = JSON.parse(stored)
        data.symptoms = answers
        
        // Triage logic
        let urgency = "LOW"
        if (answers.duration === "More than 6 months" && answers.changed === "Yes") urgency = "MODERATE"
        if (answers.symptoms?.includes("Bleeding") || answers.symptoms?.includes("Pain")) urgency = "HIGH"
        if (data.prediction === "melanoma") urgency = "HIGH"
        
        data.triage = urgency
        sessionStorage.setItem(`scan_${unwrappedParams.id}`, JSON.stringify(data))
      }
      
      router.push(`/reports/${unwrappedParams.id}`)
    }
  }

  const q = questions[currentQ]

  return (
    <div className="container mx-auto px-4 py-12 max-w-2xl">
      <Button variant="ghost" asChild className="mb-6 -ml-4 text-muted-foreground">
        <Link href={`/results/${unwrappedParams.id}`}>
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Results
        </Link>
      </Button>

      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Symptom Questionnaire</h1>
        <p className="text-muted-foreground">Adding context helps improve the screening recommendation.</p>
        <div className="flex gap-2 mt-6">
          {questions.map((_, idx) => (
            <div 
              key={idx} 
              className={`h-2 flex-1 rounded-full ${idx <= currentQ ? 'bg-primary' : 'bg-white/10'}`}
            />
          ))}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">{q.question}</CardTitle>
          {q.multi && <CardDescription>Select all that apply</CardDescription>}
        </CardHeader>
        <CardContent className="space-y-3">
          {q.options.map((opt) => {
            const isSelected = q.multi 
              ? answers[q.id]?.includes(opt) 
              : answers[q.id] === opt
              
            return (
              <div 
                key={opt}
                onClick={() => handleSelect(opt)}
                className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                  isSelected 
                    ? 'bg-primary/20 border-primary text-white' 
                    : 'bg-white/5 border-white/10 hover:border-white/30 text-muted-foreground'
                }`}
              >
                {opt}
              </div>
            )
          })}
          
          <div className="pt-6 flex justify-end">
            <Button onClick={handleNext} disabled={!answers[q.id] || (q.multi && answers[q.id].length === 0)}>
              {currentQ === questions.length - 1 ? 'Finish & View Report' : 'Next Question'}
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
