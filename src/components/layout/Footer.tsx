import Link from "next/link"
import { Activity } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-background/50 backdrop-blur-xl py-12 mt-24">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-1">
            <Link href="/" className="flex items-center gap-2 text-xl font-bold tracking-tighter text-white mb-4">
              <Activity className="h-6 w-6 text-primary" />
              <span>DERMA<span className="text-primary">AI</span></span>
            </Link>
            <p className="text-sm text-muted-foreground">
              AI-Assisted Dermatological Screening
            </p>
          </div>
          
          <div className="flex flex-col gap-2">
            <h4 className="font-semibold text-white">Platform</h4>
            <Link href="/screening" className="text-sm text-muted-foreground hover:text-white transition-colors">Screening</Link>
            <Link href="/diseases" className="text-sm text-muted-foreground hover:text-white transition-colors">Diseases</Link>
            <Link href="/#how-it-works" className="text-sm text-muted-foreground hover:text-white transition-colors">How It Works</Link>
          </div>
          
          <div className="flex flex-col gap-2">
            <h4 className="font-semibold text-white">Legal</h4>
            <Link href="/privacy" className="text-sm text-muted-foreground hover:text-white transition-colors">Privacy</Link>
            <Link href="/terms" className="text-sm text-muted-foreground hover:text-white transition-colors">Terms</Link>
            <Link href="/contact" className="text-sm text-muted-foreground hover:text-white transition-colors">Contact</Link>
          </div>
        </div>
        
        <div className="mt-12 pt-8 border-t border-white/5">
          <div className="bg-primary/10 border border-primary/20 rounded-lg p-4 mb-8">
            <p className="text-sm text-primary font-medium text-center">
              <strong>Disclaimer:</strong> DermaAI is an AI-assisted screening and educational prototype. It does not provide a confirmed medical diagnosis or replace professional medical advice.
            </p>
          </div>
          <p className="text-sm text-center text-muted-foreground">
            &copy; {new Date().getFullYear()} DermaAI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
