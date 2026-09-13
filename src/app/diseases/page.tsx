import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Search, ChevronRight } from "lucide-react"

export const diseases = [
  { slug: "melanoma", name: "Melanoma", category: "Pigmented / Lesion Conditions", aiSupported: true },
  { slug: "melanocytic-nevus", name: "Melanocytic nevus", category: "Pigmented / Lesion Conditions", aiSupported: true },
  { slug: "basal-cell-carcinoma", name: "Basal cell carcinoma", category: "Pigmented / Lesion Conditions", aiSupported: true },
  { slug: "actinic-keratosis", name: "Actinic keratosis / Bowen's disease", category: "Pigmented / Lesion Conditions", aiSupported: true },
  { slug: "benign-keratosis", name: "Benign keratosis", category: "Pigmented / Lesion Conditions", aiSupported: true },
  { slug: "eczema", name: "Eczema / Atopic dermatitis", category: "Inflammatory Conditions", aiSupported: false },
  { slug: "psoriasis", name: "Psoriasis", category: "Inflammatory Conditions", aiSupported: false },
]

export default function DiseasesLibrary() {
  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl">
      <div className="mb-12">
        <h1 className="text-3xl font-bold mb-4">Disease Library</h1>
        <p className="text-muted-foreground mb-8">
          Educational information about visible dermatological conditions.
        </p>
        
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <Input 
            placeholder="Search conditions..." 
            className="pl-10 glass border-white/20 focus:border-primary/50 text-white"
          />
        </div>
      </div>

      <div className="space-y-12">
        {/* We can group by category in a real app, keeping it simple for prototype */}
        <div>
          <h2 className="text-xl font-bold mb-4 text-white border-b border-white/10 pb-2">All Conditions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {diseases.map((d) => (
              <Link href={`/diseases/${d.slug}`} key={d.slug}>
                <Card className="h-full glass-card group">
                  <CardHeader className="pb-3">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-lg">{d.name}</CardTitle>
                      <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-white transition-colors" />
                    </div>
                    <CardDescription>{d.category}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {d.aiSupported ? (
                      <span className="inline-flex items-center rounded-full bg-primary/20 px-2.5 py-0.5 text-xs font-semibold text-primary">
                        AI Supported
                      </span>
                    ) : (
                      <span className="inline-flex items-center rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-semibold text-muted-foreground">
                        Educational Information
                      </span>
                    )}
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
