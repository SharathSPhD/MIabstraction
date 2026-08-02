import { NextResponse } from "next/server";
import showcase from "@/lib/showcase.json";

export async function GET() {
  return NextResponse.json(showcase);
}
