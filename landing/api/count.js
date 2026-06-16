import { kv } from '@vercel/kv';

export default async function handler(req) {
  if (req.method === 'POST') {
    const count = await kv.incr('dyadcore_page_views');
    return Response.json({ count });
  }
  const count = (await kv.get('dyadcore_page_views')) || 0;
  return Response.json({ count });
}

export const config = { runtime: 'edge' };
