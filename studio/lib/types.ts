export interface Capability {
  capability: string;
  kind: string;
  strategy: string;
  execution?: {
    autotune?: {
      target_met: boolean;
      scale?: {
        gap?: number;
      };
    };
  };
  behavioural_gate?: {
    result?: {
      margin_before?: number;
      margin_after?: number;
    };
  };
}

export interface Expectation {
  expectation: string;
  kind: string;
  passed: boolean;
  evidence: string;
  detail?: string;
}

export interface BuildReport {
  id?: string;
  app: string;
  base_model: string;
  passed: boolean;
  wall_clock_s: number;
  capabilities: Capability[];
  expectations: Expectation[];
  verified_against_recitation_of?: number;
  expectations_passed?: number;
}

export interface BuildStatus {
  build_id: string;
  status: "queued" | "running" | "completed" | "failed";
  report_ready?: boolean;
  report_url?: string;
}

export interface ExplainResponse {
  ok: boolean;
  explain?: string;
  detail?: string;
}

export interface BuildResponse {
  ok: boolean;
  build_id?: string;
  status?: string;
  detail?: string;
}
