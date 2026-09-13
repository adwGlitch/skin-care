import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, ShieldCheck, Microscope, Scan, FileText, Stethoscope, ChevronRight, CheckCircle2 } from "lucide-react"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-24 lg:py-32 flex flex-col items-center justify-center min-h-[90vh]">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background"></div>
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-full max-w-3xl h-64 bg-primary/20 blur-[120px] rounded-full pointer-events-none"></div>
        <div className="container relative z-10 mx-auto px-4 text-center animate-in fade-in slide-in-from-bottom-8 duration-1000">
          <div className="inline-flex items-center rounded-full border border-primary/40 bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary mb-8 shadow-[0_0_15px_rgba(0,212,255,0.15)]">
            <span className="flex h-2 w-2 rounded-full bg-primary mr-3 animate-pulse shadow-[0_0_5px_rgba(0,212,255,0.8)]"></span>
            Prototype / Demo AI
          </div>
          <h1 className="text-5xl md:text-7xl font-black tracking-tighter mb-6 leading-tight">
            Understand Your Skin.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-600 drop-shadow-sm">Early. Intelligently.</span>
          </h1>
          <p className="mx-auto max-w-2xl text-lg md:text-xl text-muted-foreground/90 mb-10 font-light leading-relaxed">
            AI-assisted preliminary screening for visible dermatological conditions &mdash; designed to help users understand their skin and decide when professional evaluation may be appropriate.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Button size="lg" asChild className="w-full sm:w-auto h-14 px-8 text-lg rounded-full animate-pulse-glow border border-primary/50">
              <Link href="/screening">
                Analyze Skin Image <ChevronRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild className="w-full sm:w-auto h-14 px-8 text-lg rounded-full border-white/20 hover:bg-white/10 glass transition-all">
              <Link href="#how-it-works">How It Works</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Why DermaAI Section */}
      <section className="py-20 bg-background/50 relative">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Why DermaAI?</h2>
            <p className="text-muted-foreground max-w-xl mx-auto">Advanced artificial intelligence designed with medical trust and explainability in mind.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: Scan, title: "AI-Assisted Screening", desc: "State-of-the-art vision models for preliminary assessment." },
              { icon: Microscope, title: "Explainable Results", desc: "Heatmap visualizations showing exactly what the AI sees." },
              { icon: Activity, title: "Confidence & Uncertainty", desc: "Transparent probabilistic outputs, not black-box guesses." },
              { icon: FileText, title: "Symptom-Aware Analysis", desc: "Contextual understanding combined with visual data." },
              { icon: ShieldCheck, title: "Longitudinal Tracking", desc: "Track skin changes over time with timeline comparisons." },
              { icon: Stethoscope, title: "Professional Referral", desc: "Clear guidance on when to seek clinical evaluation." },
            ].map((feature, i) => (
              <Card key={i} className="glass-card shadow-lg shadow-black/20 group">
                <CardHeader>
                  <div className="h-14 w-14 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <feature.icon className="h-7 w-7 text-primary" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-base font-light">{feature.desc}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 relative">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground max-w-xl mx-auto">A simple, transparent process from capture to assessment.</p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { step: "01", title: "Upload" },
              { step: "02", title: "Quality Check" },
              { step: "03", title: "AI Analysis" },
              { step: "04", title: "Explain" },
              { step: "05", title: "Review" },
              { step: "06", title: "Next Step" },
            ].map((item, i) => (
              <div key={i} className="flex flex-col items-center text-center p-6 relative">
                <div className="text-4xl font-black text-white/5 mb-3">{item.step}</div>
                <div className="font-semibold text-white/90">{item.title}</div>
                {i < 5 && <ChevronRight className="hidden lg:block absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 text-white/10 h-8 w-8" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Safety & Disclaimer */}
      <section className="py-20 bg-primary/5 border-y border-primary/10">
        <div className="container mx-auto px-4 text-center max-w-3xl">
          <ShieldCheck className="h-12 w-12 text-primary mx-auto mb-6" />
          <h2 className="text-2xl font-bold mb-4">Safety First</h2>
          <p className="text-lg text-muted-foreground mb-8">
            DermaAI provides preliminary AI-assisted information and is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
          </p>
          <Button size="lg" asChild className="rounded-full">
            <Link href="/screening">Start Your First Screening</Link>
          </Button>
        </div>
      </section>
    </div>
  )
}
