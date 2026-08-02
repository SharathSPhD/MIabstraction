export interface Capability {
  capability: string;
  kind: string;
  clause?: string;
  strategy: string;
  reason?: string;
  execution?: {
    autotune?: {
      skipped?: boolean;
      scale?: {
        instructed_cost?: number;
        uninstructed_cost?: number;
        gap?: number;
        must_recover?: number;
        target_nats?: number;
      };
      trials?: Array<{
        config: Record<string, unknown>;
        score: number;
        rejected_reason?: string;
      }>;
      n_trials?: number;
      n_admissible?: number;
      target_met?: boolean;
    };
  };
  behavioural_gate?: {
    result?: {
      margin_before?: number;
      margin_after?: number;
      target?: number;
      rates?: {
        before?: number;
        after?: number;
      };
    };
    margin?: number;
    budget?: number;
    resolution?: number;
    note?: string;
    trials?: Array<{
      config: Record<string, unknown>;
      score: number;
      rejected_reason?: string;
    }>;
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
  program_id?: string;
  capabilities: Capability[];
  expectations: Expectation[];
  verified_against_recitation_of?: number;
  expectations_passed?: number;
  side_effect_guard?: {
    budget?: number;
    resolution?: number;
    note?: string;
  };
  search_space?: {
    explained?: string;
  };
  controls?: Array<{
    name?: string;
    layer?: string;
    strength?: number;
    side_effect?: number;
  }>;
  n_controls_installed?: number;
  execution?: {
    adapter_saved_to?: string[];
  };
  hf_repo?: string;
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
