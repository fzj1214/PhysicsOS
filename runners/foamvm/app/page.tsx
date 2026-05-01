import { HomePage } from '@/components/home-page'
import { getViewerContext } from '@/lib/auth'

export default async function Page() {
  const viewer = await getViewerContext()

  return <HomePage viewer={viewer} />
}
