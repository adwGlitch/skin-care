import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Plus, History, Activity, ShieldCheck, Clock, FileText } from "lucide-react"

export default function DashboardPage() {
  return (
    <div className="container mx-auto px-4 py-12 max-w-6xl">
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold">Welcome back, User</h1>
          <p className="text-muted-foreground mt-1">Here is a summary of your skin health tracking.</p>
        </div>
        <Button asChild className="rounded-full">
          <Link href="/screening"><Plus className="mr-2 h-4 w-4" /> New Skin Screening</Link>
        </Button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-10">
        {[
          { title: "Total Scans", value: "3", icon: History },
          { title: "Recent Scans", value: "1", icon: Activity },
          { title: "Tracked Lesions", value: "2", icon: ShieldCheck },
          { title: "Generated Reports", value: "3", icon: FileText },
        ].map((stat, i) => (
          <Card key={i} className="glass-card">
            <CardContent className="p-6 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{stat.title}</p>
                <p className="text-3xl font-black mt-2 text-white">{stat.value}</p>
              </div>
              <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center border border-primary/20">
                <stat.icon className="h-6 w-6 text-primary" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Analyses */}
        <div className="lg:col-span-2">
          <Card className="glass">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Recent Analyses</CardTitle>
                <CardDescription>Your latest AI screening results</CardDescription>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/history">View All</Link>
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { date: "Today", condition: "Possible melanocytic nevus", confidence: "92%", status: "Preliminary", id: "scan_demo_1" },
                  { date: "12 Sep 2026", condition: "Possible vascular lesion", confidence: "87%", status: "Preliminary", id: "scan_demo_2" },
                  { date: "15 Aug 2026", condition: "Possible benign keratosis", confidence: "95%", status: "Preliminary", id: "scan_demo_3" }
                ].map((scan, i) => (
                  <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
                    <div className="flex items-start sm:items-center gap-4 mb-4 sm:mb-0">
                      <div className="h-10 w-10 rounded bg-primary/20 flex items-center justify-center shrink-0">
                        <Activity className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{scan.condition}</p>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                          <span className="flex items-center"><Clock className="mr-1 h-3 w-3" /> {scan.date}</span>
                          <span className="bg-white/10 px-2 py-0.5 rounded">{scan.confidence} Confidence</span>
                          <span className="text-primary">{scan.status}</span>
                        </div>
                      </div>
                    </div>
                    <Button variant="secondary" size="sm" asChild>
                      <Link href={`/results/${scan.id}`}>View Result</Link>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions & Tracked Lesions */}
        <div className="space-y-8">
          <Card className="glass shadow-lg border-primary/30 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 blur-[50px] -mr-16 -mt-16 pointer-events-none"></div>
            <CardHeader>
              <CardTitle>Tracked Lesions</CardTitle>
              <CardDescription>Monitor changes over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 mb-6">
                {[
                  { name: "Left forearm lesion", location: "Left arm", updates: 2 },
                  { name: "Back shoulder spot", location: "Back", updates: 1 }
                ].map((lesion, i) => (
                  <div key={i} className="flex justify-between items-center p-3 rounded-lg bg-black/20 border border-white/5">
                    <div>
                      <p className="font-medium text-sm text-white">{lesion.name}</p>
                      <p className="text-xs text-muted-foreground">{lesion.location} • {lesion.updates} scans</p>
                    </div>
                    <Button variant="ghost" size="sm">View</Button>
                  </div>
                ))}
              </div>
              <Button className="w-full" variant="outline" asChild>
                <Link href="/tracking">Manage Tracked Lesions</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
