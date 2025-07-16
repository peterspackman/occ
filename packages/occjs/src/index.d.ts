/**
 * TypeScript definitions for OCC JavaScript/WebAssembly bindings
 */

export interface Vec3 {
  x(): number;
  y(): number;
  z(): number;
  setX(val: number): void;
  setY(val: number): void;
  setZ(val: number): void;
}

export interface IVec3 {
  0: number;
  1: number;
  2: number;
}

export interface Mat3 {
  rows(): number;
  cols(): number;
  get(row: number, col: number): number;
  set(row: number, col: number, val: number): void;
  static create(rows: number, cols: number): Mat3;
}

export interface Mat3N {
  set(row: number, col: number, val: number): void;
  get(row: number, col: number): number;
  rows(): number;
  cols(): number;
  static create(cols: number): Mat3N;
}

export interface IVec {
  size(): number;
  get(i: number): number;
  set(i: number, val: number): void;
  static fromArray(array: number[]): IVec;
}

export interface Vec {
  size(): number;
  get(i: number): number;
  set(i: number, val: number): void;
  static create(size: number): Vec;
}

export interface Mat {
  rows(): number;
  cols(): number;
  get(row: number, col: number): number;
  set(row: number, col: number, val: number): void;
  static create(rows: number, cols: number): Mat;
}

export interface Element {
  symbol: string;
  mass: number;
  name: string;
  vanDerWaalsRadius: number;
  covalentRadius: number;
  atomicNumber: number;
  toString(): string;
  static fromAtomicNumber(atomicNumber: number): Element;
}

export interface Atom {
  atomicNumber: number;
  x: number;
  y: number;
  z: number;
  getPosition(): Vec3;
  setPosition(pos: Vec3): void;
  toString(): string;
}

export interface PointCharge {
  charge: number;
  getPosition(): Vec3;
  setCharge(charge: number): void;
  setPosition(pos: Vec3): void;
  toString(): string;
}

export enum Origin {
  CARTESIAN = 0,
  CENTROID = 1,
  CENTEROFMASS = 2
}

export interface Molecule {
  size(): number;
  elements(): IVec;
  positions(): Mat3N;
  name(): string;
  setName(name: string): void;
  partialCharges(): Vec;
  setPartialCharges(charges: Vec): void;
  espPartialCharges(): Vec;
  atomicMasses(): Vec;
  atomicNumbers(): IVec;
  vdwRadii(): Vec;
  molarMass(): number;
  atoms(): Atom[];
  centerOfMass(): Vec3;
  centroid(): Vec3;
  rotate(rotationMatrix: Mat3, origin: Origin): void;
  translate(translation: Vec3): void;
  rotated(rotationMatrix: Mat3, origin: Origin): Molecule;
  translated(translation: Vec3): Molecule;
  centered(origin: Origin): Molecule;
  translationalFreeEnergy(temperature: number, pressure: number): number;
  rotationalFreeEnergy(temperature: number): number;
  toString(): string;
  static fromXyzFile(filename: string): Molecule;
  static fromXyzString(contents: string): Molecule;
}

export interface Dimer {
  a: Molecule;
  b: Molecule;
  nearestDistance: number;
  centerOfMassDistance: number;
  centroidDistance: number;
  symmetryRelation(): string;
  name: string;
  setName(name: string): void;
}

export enum PointGroup {
  C1 = 0,
  Ci = 1,
  Cs = 2,
  C2 = 3,
  C3 = 4,
  C4 = 5,
  C5 = 6,
  C6 = 7,
  C2v = 8,
  C3v = 9,
  C4v = 10,
  C5v = 11,
  C6v = 12,
  D2 = 13,
  D3 = 14,
  D4 = 15,
  D5 = 16,
  D6 = 17,
  D2h = 18,
  D3h = 19,
  D4h = 20,
  D5h = 21,
  D6h = 22,
  Td = 23,
  Oh = 24
}

export interface MolecularPointGroup {
  getDescription(): string;
  getPointGroupString(): string;
  pointGroup: PointGroup;
  symmetryNumber: number;
  toString(): string;
}

export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  CRITICAL = 5,
  OFF = 6
}

export enum SpinorbitalKind {
  Restricted = 0,
  Unrestricted = 1,
  General = 2
}

export type LogCallback = (level: LogLevel, message: string) => void;

export interface LogEntry {
  level: number;
  message: string;
}

// Quantum mechanics types
export interface AOBasis {
  nbf(): number;
  nao(): number;
  nsh(): number;
  atomOffsets(): IVec;
  shellOffsets(): IVec;
  firstBasisFunctionOfShell(shell: number): number;
  static loadFromJsonFile(molecule: Molecule, filename: string): AOBasis;
  static loadFromJsonString(molecule: Molecule, jsonString: string): AOBasis;
}

export interface MolecularOrbitals {
  C: Mat;
  Cocc: Mat;
  Cvirt: Mat;
  energies: Vec;
  energiesOcc: Vec;
  energiesVirt: Vec;
  nAlpha: number;
  nBeta: number;
  nElectrons: number;
  nAOs: number;
  nMOs: number;
}

export interface HartreeFock {
  overlap(): Mat;
  kinetic(): Mat;
  nuclear(): Mat;
  coulomb(D: Mat): Mat;
  exchange(D: Mat): Mat;
  fock(D: Mat): Mat;
}

export interface SCFConvergenceSettings {
  energyThreshold: number;
  densityThreshold: number;
  maxIterations: number;
  diisMaxVectors: number;
  diisStartIteration: number;
}

export interface HartreeFockSCF {
  setConvergenceSettings(settings: SCFConvergenceSettings): void;
  run(): void;
  energy(): number;
  converged(): boolean;
  iterations(): number;
  mo(): MolecularOrbitals;
}

// Cube file interface
export interface Cube {
  name: string;
  description: string;
  getOrigin(): Vec3;
  setOrigin(x: number, y: number, z: number): void;
  getBasis(): Mat3;
  setBasis(basis: Mat3): void;
  getSteps(): IVec3;
  setSteps(nx: number, ny: number, nz: number): void;
  centerMolecule(): void;
  fillElectronDensity(mol: Molecule, wfn: Wavefunction): void;
  fillPromoleculeDensity(mol: Molecule): void;
  fillElectricPotential(mol: Molecule, wfn: Wavefunction): void;
  getData(): Float32Array;
  setData(data: Float32Array): void;
  saveToString(): string;
  static loadFromString(content: string): Cube;
}

// Isosurface enums
export enum SurfaceKind {
  ElectronDensity = 0,
  PromoleculeDensity = 1,
  Orbital = 2,
  ElectricPotential = 3,
  DeformationDensity = 4,
  SpinDensity = 5
}

export enum PropertyKind {
  ElectronDensity = 0,
  ElectricPotential = 1,
  Orbital = 2,
  DeformationDensity = 3,
  SpinDensity = 4
}

// Isosurface types
export interface OrbitalIndex {
  offset: number;
  reference: OrbitalReference;
}

export enum OrbitalReference {
  Absolute = 0,
  HOMO = 1,
  LUMO = 2
}

export interface IsosurfaceParameters {
  isovalue: number;
  separation: number;
  surfaceKind: SurfaceKind;
  flipNormals: boolean;
  properties: PropertyKind[];
}

export interface MeshData {
  vertices: Float32Array;
  faces: Uint32Array;
  normals: Float32Array;
  numVertices: number;
  numFaces: number;
  volume: number;
  surfaceArea: number;
}

export interface Isosurface {
  isovalue: number;
  separation: number;
  kind: string;
  description: string;
  volume(): number;
  surfaceArea(): number;
  getVertices(): Float32Array;
  getFaces(): Uint32Array;
  getNormals(): Float32Array;
  getMeshData(): MeshData;
}

export interface IsosurfaceCalculator {
  setMolecule(mol: Molecule): void;
  setWavefunction(wfn: Wavefunction): void;
  setParameters(params: IsosurfaceParameters): void;
  validate(): boolean;
  compute(): void;
  getIsosurface(): Isosurface;
}

// Wavefunction interface
export interface Wavefunction {
  molecularOrbitals: MolecularOrbitals;
  atoms: Atom[];
  basis: AOBasis;
  numAlphaElectrons: number;
  numBetaElectrons: number;
  energy: number;
  translate(translation: Vec3): void;
  transform(matrix: Mat3): void;
  charge(): number;
  static load(filename: string): Wavefunction;
  save(filename: string): void;
}

// Main module interface
export interface OCCModule {
  // Math types
  Vec3: typeof Vec3;
  Mat3N: typeof Mat3N;
  IVec: typeof IVec;
  Vec: typeof Vec;
  Mat: typeof Mat;
  
  // Core types
  Element: typeof Element;
  Atom: typeof Atom;
  PointCharge: typeof PointCharge;
  Molecule: typeof Molecule;
  Dimer: typeof Dimer;
  MolecularPointGroup: typeof MolecularPointGroup;
  
  // Enums
  Origin: typeof Origin;
  PointGroup: typeof PointGroup;
  LogLevel: typeof LogLevel;
  
  // QM types
  AOBasis: typeof AOBasis;
  MolecularOrbitals: typeof MolecularOrbitals;
  HartreeFock: typeof HartreeFock;
  SCFConvergenceSettings: typeof SCFConvergenceSettings;
  HartreeFockSCF: typeof HartreeFockSCF;
  
  // Utility functions
  eemPartialCharges(atomicNumbers: IVec, positions: Mat3N, charge?: number): Vec;
  eeqPartialCharges(atomicNumbers: IVec, positions: Mat3N, charge?: number): Vec;
  eeqCoordinationNumbers(atomicNumbers: IVec, positions: Mat3N): Vec;
  
  // Data directory functions
  setDataDirectory(path: string): void;
  getDataDirectory(): string;
  
  // Logging functions
  setLogLevel(level: number): void;
  setLogLevelString(level: string): void;
  registerLogCallback(callback: LogCallback): void;
  clearLogCallbacks(): void;
  getBufferedLogs(): LogEntry[];
  clearLogBuffer(): void;
  setLogBuffering(enable: boolean): void;
  setLogFile(filename: string): void;
  
  // Direct logging
  logTrace(message: string): void;
  logDebug(message: string): void;
  logInfo(message: string): void;
  logWarn(message: string): void;
  logError(message: string): void;
  logCritical(message: string): void;
  
  // Other utilities
  setNumThreads(n: number): void;
  version: string;
  
  // Isosurface types
  Cube: typeof Cube;
  SurfaceKind: typeof SurfaceKind;
  PropertyKind: typeof PropertyKind;
  OrbitalIndex: typeof OrbitalIndex;
  IsosurfaceParameters: typeof IsosurfaceParameters;
  Isosurface: typeof Isosurface;
  IsosurfaceCalculator: typeof IsosurfaceCalculator;
  
  // Isosurface helper functions
  generateElectronDensityIsosurface(
    wfn: Wavefunction,
    isovalue: number,
    separation: number
  ): MeshData;
  
  generatePromoleculeDensityIsosurface(
    mol: Molecule,
    isovalue: number,
    separation: number
  ): MeshData;
  
  // JSON export
  isosurfaceToJSON(surf: Isosurface): string;
}

export interface LoadOptions {
  wasmPath?: string;
  env?: Record<string, any>;
}

export declare function loadOCC(options?: LoadOptions): Promise<OCCModule>;
export declare function moleculeFromXYZ(xyzString: string): Promise<Molecule>;
export declare function createMolecule(atomicNumbers: number[], positions: number[][]): Promise<Molecule>;

export declare const Elements: Record<string, number>;
export declare const BasisSets: Record<string, string>;
export declare const Module: OCCModule;