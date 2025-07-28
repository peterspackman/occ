/**
 * TypeScript definitions for DMA (Distributed Multipole Analysis) functionality
 */

import { Molecule } from './index.d.ts';

// DMA-specific interfaces that should be added to the main module
export interface Mult {
  max_rank: number;
  q: Vec;
  
  // Level 0 (monopole)
  Q00(): number;
  charge(): number;
  
  // Level 1 (dipole)
  Q10(): number;
  Q11c(): number;
  Q11s(): number;
  
  // Level 2 (quadrupole)
  Q20(): number;
  Q21c(): number;
  Q21s(): number;
  Q22c(): number;
  Q22s(): number;
  
  // Level 3 (octupole)
  Q30(): number;
  Q31c(): number;
  Q31s(): number;
  Q32c(): number;
  Q32s(): number;
  Q33c(): number;
  Q33s(): number;
  
  // Level 4 (hexadecapole)
  Q40(): number;
  Q41c(): number;
  Q41s(): number;
  Q42c(): number;
  Q42s(): number;
  Q43c(): number;
  Q43s(): number;
  Q44c(): number;
  Q44s(): number;
  
  num_components(): number;
  toString(rank: number): string;
}

export interface DMASettings {
  max_rank: number;
  big_exponent: number;
  include_nuclei: boolean;
}

export interface DMAResultInterface {
  max_rank: number;
  multipoles: Mult[];
}

export interface DMASites {
  size(): number;
  num_atoms(): number;
  atoms: unknown[];
  name: string[];
  positions: Mat3N;
  atom_indices: IVec;
  radii: Vec;
  limits: IVec;
}

export interface DMACalculator {
  update_settings(settings: DMASettings): void;
  settings(): DMASettings;
  set_radius_for_element(atomic_number: number, radius_angs: number): void;
  set_limit_for_element(atomic_number: number, limit: number): void;
  sites(): DMASites;
  compute_multipoles(): DMAResultInterface;
  compute_total_multipoles(result: DMAResultInterface): Mult;
}

// High-level JavaScript API
export interface DMAOptions {
  maxRank?: number;
  bigExponent?: number;
  includeNuclei?: boolean;
  atomRadii?: Record<string, number>;
  atomLimits?: Record<string, number>;
}

export interface MultipoleComponents {
  Q00: number;
  charge: number;
  Q10: number;
  Q11c: number;
  Q11s: number;
  Q20: number;
  Q21c: number;
  Q21s: number;
  Q22c: number;
  Q22s: number;
  maxRank: number;
}

export declare class DMAConfig {
  maxRank: number;
  bigExponent: number;
  includeNuclei: boolean;
  atomRadii: Map<string, number>;
  atomLimits: Map<string, number>;
  writePunch: boolean;
  punchFilename: string;
  
  constructor();
  setAtomRadius(element: string, radius: number): void;
  setAtomLimit(element: string, maxRank: number): void;
}

export declare class DMAResult {
  result: DMAResultInterface;
  sites: DMASites;
  total: Mult;
  
  constructor(result: DMAResultInterface, sites: DMASites, total: Mult);
  getSiteMultipoles(siteIndex: number): Mult;
  getTotalMultipoles(): Mult;
  getSites(): DMASites;
  toPunchFile(): string;
  getMultipoleComponent(siteIndex: number, component: string): number;
  getAllComponents(siteIndex: number): MultipoleComponents;
}

export declare function calculateDMA(
  molecule: Molecule,
  basisName: string,
  options?: DMAOptions
): Promise<DMAResult>;

export declare function generatePunchFile(dmaResult: DMAResult): string;

declare const DMAModule: {
  DMAConfig: typeof DMAConfig;
  DMAResult: typeof DMAResult;
  calculateDMA: typeof calculateDMA;
  generatePunchFile: typeof generatePunchFile;
};

export default DMAModule;