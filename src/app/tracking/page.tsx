"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Plus, ArrowRight, User } from "lucide-react"

const trackedLesions = [
  {
    id: "1",
    name: "Left forearm lesion",
    location: "Left arm",
    scans: [
      { date: "12 Sep 2026", img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=200", change: "None" },
      { date: "10 Aug 2026", img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=200", change: "Initial" }
    ]
  },
  {
    id: "2",
    name: "Back shoulder spot",
    location: "Back",
    scans: [
      { date: "15 Aug 2026", img: "https://images.unsplash.com/photo-1512495039889-52a3b799c9bc?auto=format&fit=crop&q=80&w=200", change: "Initial" }
    ]
  }
]

export default function TrackingPage() {
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null)

  const bodyParts = [
    "Head", "Face", "Neck", "Chest", "Back", "Left arm", "Right arm", 
    "Left hand", "Right hand", "Left leg", "Right leg", "Left foot", "Right foot"
  ]

  return (
    <div className="container mx-auto px-4 py-12 max-w-6xl">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold">Lesion Tracking</h1>
          <p className="text-muted-foreground mt-1">Monitor changes in your skin over time.</p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" /> Track New Lesion
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Body Map */}
        <div className="lg:col-span-1">
          <Card className="h-full glass">
            <CardHeader>
              <CardTitle>Body Map</CardTitle>
              <CardDescription>Filter lesions by location</CardDescription>
            </CardHeader>
            <CardContent>
              {/* Conceptual Interactive Body Map */}
              <div className="aspect-[1/2] w-full max-w-[200px] mx-auto bg-black/40 rounded-3xl border border-white/10 flex items-center justify-center relative mb-8 shadow-inner">
                <User className="h-32 w-32 text-white/10" strokeWidth={1} />
                {/* Mock hotspots */}
                <div className="absolute top-1/3 left-1/4 w-3 h-3 bg-primary rounded-full shadow-[0_0_10px_rgba(0,212,255,0.8)] animate-pulse" />
                <div className="absolute top-1/4 right-1/3 w-3 h-3 bg-primary rounded-full shadow-[0_0_10px_rgba(0,212,255,0.8)]" />
              </div>
              
              <div className="flex flex-wrap gap-2">
                <Button 
                  variant={selectedLocation === null ? "default" : "secondary"}
                  size="sm"
                  onClick={() => setSelectedLocation(null)}
                  className="rounded-full text-xs h-7"
                >
                  All Locations
                </Button>
                {bodyParts.map(part => (
                  <Button 
                    key={part}
                    variant={selectedLocation === part ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setSelectedLocation(part)}
                    className="rounded-full text-xs h-7 border border-white/10 bg-white/5"
                  >
                    {part}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Tracked Lesions Timeline */}
        <div className="lg:col-span-2 space-y-6">
          {trackedLesions
            .filter(l => selectedLocation === null || l.location === selectedLocation)
            .map(lesion => (
            <Card key={lesion.id} className="glass">
              <CardHeader className="flex flex-row items-start justify-between pb-4 border-b border-white/5 mb-4">
                <div>
                  <CardTitle className="text-xl">{lesion.name}</CardTitle>
                  <CardDescription className="flex items-center gap-2 mt-1">
                    <span className="bg-white/10 px-2 py-0.5 rounded text-xs">{lesion.location}</span>
                    <span>Tracked since {lesion.scans[lesion.scans.length-1].date}</span>
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm">
                  <Plus className="mr-2 h-4 w-4" /> Add Scan
                </Button>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4 overflow-x-auto pb-4 pt-2 hide-scrollbar">
                  {lesion.scans.map((scan, i) => (
                    <div key={i} className="flex items-center">
                      <div className="flex flex-col gap-2 min-w-[140px]">
                        <div className="relative aspect-square rounded-lg overflow-hidden border-2 border-white/10">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={scan.img} alt={`Scan on ${scan.date}`} className="w-full h-full object-cover" />
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-white">{scan.date}</p>
                          <p className="text-xs text-muted-foreground">{scan.change}</p>
                        </div>
                      </div>
                      
                      {i < lesion.scans.length - 1 && (
                        <div className="px-4 text-white/20">
                          <ArrowRight className="h-6 w-6" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                
                {lesion.scans.length > 1 && (
                  <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <p className="text-sm text-blue-400 font-medium">
                      Visual difference detected. No significant changes in size or color pattern.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
          
          {trackedLesions.filter(l => selectedLocation === null || l.location === selectedLocation).length === 0 && (
            <div className="text-center py-12 border border-white/10 rounded-xl bg-white/5">
              <p className="text-muted-foreground">No tracked lesions found for this location.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
