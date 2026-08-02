import { NextResponse } from "next/server";
import examples from "@/lib/examples.json";

export async function GET() {
  return NextResponse.json(examples);
}
