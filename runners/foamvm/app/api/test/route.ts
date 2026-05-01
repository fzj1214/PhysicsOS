export async function GET() {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      for (let i = 1; i <= 3; i++) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'status', message: `ping ${i}` })}\n\n`)
        )
        await new Promise((r) => setTimeout(r, 500))
      }
      controller.close()
    },
  })
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}
