import { use } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ShieldAlert, ArrowLeft, Scan } from "lucide-react"

// In a real app this would be fetched from a database
const diseaseData: Record<string, any> = {
  "melanoma": {
    name: "Melanoma",
    category: "Pigmented / Lesion Conditions",
    overview: "Melanoma is the most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color.",
    characteristics: [
      "A large brownish spot with darker speckles",
      "A mole that changes in color, size or feel or that bleeds",
      "A small lesion with an irregular border and portions that appear red, pink, white, blue or blue-black",
      "A painful lesion that itches or burns"
    ],
    symptoms: ["Itching", "Bleeding", "Oozing", "Change in size/shape"],
    whenToSeek: "Immediate professional evaluation is recommended if you notice any new, unusual, or changing moles or spots on your skin.",
    risk: "High risk of spreading if not caught early.",
    aiSupported: true
  }
}

export default function DiseaseDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const unwrappedParams = use(params)
  const data = diseaseData[unwrappedParams.slug] || {
    name: unwrappedParams.slug.replace("-", " "),
    category: "Unknown",
    overview: "Educational information about this condition is currently being updated.",
    characteristics: [],
    symptoms: [],
    whenToSeek: "Consult a healthcare professional for evaluation.",
    risk: "Unknown",
    aiSupported: false
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <Button variant="ghost" asChild className="mb-6 -ml-4 text-muted-foreground">
        <Link href="/diseases">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Library
        </Link>
      </Button>

      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <h1 className="text-4xl font-bold capitalize">{data.name}</h1>
          {data.aiSupported && (
            <span className="inline-flex items-center rounded-full bg-primary/20 px-2.5 py-0.5 text-xs font-semibold text-primary">
              AI Supported
            </span>
          )}
        </div>
        <p className="text-muted-foreground text-lg">{data.category}</p>
      </div>

      <div className="bg-primary/5 border border-primary/20 p-6 rounded-xl mb-8 flex gap-4 items-start">
        <ShieldAlert className="h-6 w-6 text-primary shrink-0 mt-1" />
        <div>
          <h4 className="font-semibold text-primary mb-1">Educational information only</h4>
          <p className="text-sm text-primary/80">This page provides general educational information and does not constitute medical advice or diagnosis.</p>
        </div>
      </div>

      <div className="space-y-8">
        <section>
          <h2 className="text-2xl font-semibold mb-4 border-b border-white/10 pb-2">Overview</h2>
          <p className="text-white/90 leading-relaxed">{data.overview}</p>
        </section>

        {data.characteristics.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold mb-4 border-b border-white/10 pb-2">Common Characteristics</h2>
            <ul className="list-disc pl-5 space-y-2 text-white/90">
              {data.characteristics.map((c: string, i: number) => <li key={i}>{c}</li>)}
            </ul>
          </section>
        )}

        <section className="bg-white/5 p-6 rounded-xl border border-white/10">
          <h2 className="text-xl font-semibold mb-4 text-white">When to seek professional evaluation</h2>
          <p className="text-white/80">{data.whenToSeek}</p>
        </section>

        <div className="mt-12 flex justify-center">
          <Button size="lg" asChild className="rounded-full">
            <Link href="/screening">
              <Scan className="mr-2 h-5 w-5" /> Analyze an Image
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
