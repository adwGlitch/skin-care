import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("image")
    
    if (!file) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    const aiMode = process.env.NEXT_PUBLIC_AI_MODE || "demo"
    const aiApiUrl = process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000"

    if (aiMode === "real") {
      // Proxy to FastAPI Backend
      try {
        const backendResponse = await fetch(`${aiApiUrl}/api/analyze`, {
          method: 'POST',
          body: formData,
        })
        
        if (!backendResponse.ok) {
          const errorText = await backendResponse.text()
          console.error("FastAPI backend error:", errorText)
          return NextResponse.json({ error: "AI Service returned an error" }, { status: backendResponse.status })
        }
        
        const data = await backendResponse.json()
        return NextResponse.json(data)
      } catch (err) {
        console.error("Failed to connect to FastAPI backend:", err)
        return NextResponse.json({ error: "Could not connect to AI Service. Is the backend running?" }, { status: 503 })
      }
    }

    // ==========================================
    // DEMO MODE (Fallback)
    // ==========================================
    await new Promise((resolve) => setTimeout(resolve, 3000))

    const response = {
      status: "success",
      prediction: "mel",
      confidence: 0.82,
      top_predictions: [
        { class: "mel", probability: 0.82, name: "Melanoma" },
        { class: "nv", probability: 0.09, name: "Melanocytic nevus" },
        { class: "bcc", probability: 0.05, name: "Basal cell carcinoma" },
        { class: "other", probability: 0.04, name: "Other" }
      ],
      image_quality: {
        score: 0.91,
        status: "good",
        messages: []
      },
      uncertainty: "moderate",
      heatmap_url: "/demo-heatmap.jpg", 
      recommendation_level: "professional_evaluation",
      scan_id: Math.random().toString(36).substring(7),
      model_version: "demo-v1.0"
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("Analysis error:", error)
    return NextResponse.json({ error: "Analysis failed" }, { status: 500 })
  }
}
