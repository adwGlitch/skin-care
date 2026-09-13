import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Activity } from "lucide-react"

export function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-background/50 backdrop-blur-xl">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 text-xl font-bold tracking-tighter text-white">
          <Activity className="h-6 w-6 text-primary" />
          <span>DERMA<span className="text-primary">AI</span></span>
        </Link>
        
        <div className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground">
          <Link href="/" className="hover:text-white transition-colors">Home</Link>
          <Link href="/screening" className="hover:text-white transition-colors">AI Screening</Link>
          <Link href="/diseases" className="hover:text-white transition-colors">Diseases</Link>
          <Link href="/#how-it-works" className="hover:text-white transition-colors">How It Works</Link>
        </div>
        
        <div className="flex items-center gap-4">
          <Button variant="ghost" className="hidden sm:inline-flex text-muted-foreground hover:text-white">
            Sign In
          </Button>
          <Button asChild className="rounded-full">
            <Link href="/screening">Get Started</Link>
          </Button>
        </div>
      </div>
    </nav>
  )
}
