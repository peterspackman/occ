import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC } from '../dist/index.js';

// A model surface: a double-well bond 0-1 (minima at 1.6 and 2.4 Bohr, a
// barrier of 0.0256 Hartree at 2.0) and springs on 0-2 and 1-2.
const BOHR = 1.8897261246257702; // per Angstrom
const TERMS = [
  [0, 1, 'well', 1.0, 1.6, 2.4],
  [0, 2, 'spring', 0.5, 1.81],
  [1, 2, 'spring', 0.1, 2.86],
];

describe('Saddle point search and nudged elastic band', () => {
  let Module;

  const coordinates = (molecule) => {
    const p = molecule.positions();
    const x = [];
    for (let a = 0; a < p.cols(); a++)
      {x.push([0, 1, 2].map((k) => p.get(k, a) * BOHR));}
    return x;
  };

  const surface = (molecule) => {
    const x = coordinates(molecule);
    let energy = 0;
    const g = Module.Mat3N.create(x.length);
    for (const [i, j, kind, a, b, c] of TERMS) {
      const d = [0, 1, 2].map((k) => x[i][k] - x[j][k]);
      const r = Math.hypot(...d);
      let de;
      if (kind === 'well') {
        const u = r - b, v = r - c;
        energy += a * u * u * v * v;
        de = 2 * a * u * v * (u + v);
      } else {
        energy += 0.5 * a * (r - b) ** 2;
        de = a * (r - b);
      }
      for (let k = 0; k < 3; k++) {
        g.set(k, i, g.get(k, i) + (de * d[k]) / r);
        g.set(k, j, g.get(k, j) - (de * d[k]) / r);
      }
    }
    return [energy, g];
  };

  const triatomic = (r01, r02) => {
    const half = ((104.5 / 2) * Math.PI) / 180;
    const p = Module.Mat3N.create(3);
    p.set(0, 1, (r01 * Math.sin(half)) / BOHR);
    p.set(2, 1, (r01 * Math.cos(half)) / BOHR);
    p.set(0, 2, (-r02 * Math.sin(half)) / BOHR);
    p.set(2, 2, (r02 * Math.cos(half)) / BOHR);
    return new Module.Molecule(Module.IVec.fromArray([8, 1, 1]), p);
  };

  const bond = (molecule, i, j) => {
    const x = coordinates(molecule);
    return Math.hypot(...[0, 1, 2].map((k) => x[i][k] - x[j][k]));
  };

  const run = (optimizer, steps) => {
    for (let s = 0; s < steps; s++) {
      const [e, g] = surface(optimizer.getNextGeometry());
      optimizer.update(e, g);
      if (optimizer.step()) {return true;}
    }
    return false;
  };

  const hessian = (molecule) => {
    const p = molecule.positions();
    const n = p.cols();
    const h = 1e-4; // Bohr
    const H = Module.Mat.create(3 * n, 3 * n);
    for (let c = 0; c < 3 * n; c++) {
      const columns = [1, -1].map((sign) => {
        const q = Module.Mat3N.create(n);
        for (let k = 0; k < 3; k++)
          {for (let a = 0; a < n; a++) {q.set(k, a, p.get(k, a));}}
        q.set(c % 3, Math.floor(c / 3),
              q.get(c % 3, Math.floor(c / 3)) + (sign * h) / BOHR);
        return surface(new Module.Molecule(Module.IVec.fromArray([8, 1, 1]), q))[1];
      });
      for (let r = 0; r < 3 * n; r++) {
        const k = r % 3, a = Math.floor(r / 3);
        H.set(r, c, (columns[0].get(k, a) - columns[1].get(k, a)) / (2 * h));
      }
    }
    return H;
  };

  beforeAll(async () => {
    Module = await loadOCC();
    if (Module.setLogLevel) {Module.setLogLevel('WARN');}
  });

  it('finds the saddle point from a computed Hessian', () => {
    const start = triatomic(1.85, 1.75);
    const optimizer = new Module.GeometryOptimizer(
      start, new Module.ConvergenceCriteria(),
      Module.OptimizerSettings.saddlePoint(0));
    optimizer.setInitialHessian(hessian(start));
    expect(run(optimizer, 50)).toBe(true);
    expect(optimizer.negativeModes()).toBe(1);
    expect(bond(optimizer.getNextGeometry(), 0, 1)).toBeCloseTo(2.0, 2);
  });

  it('climbs a nudged elastic band to the saddle point', () => {
    const relaxed = (r01) => {
      const optimizer = new Module.GeometryOptimizer(triatomic(r01, 1.81));
      expect(run(optimizer, 50)).toBe(true);
      return optimizer.getNextGeometry();
    };
    const reactant = relaxed(1.6), product = relaxed(2.4);
    const settings = new Module.NebSettings();
    settings.images = 5;
    const band = new Module.NudgedElasticBand(reactant, product, settings);
    band.setEndpointEnergies(surface(reactant)[0], surface(product)[0]);
    let converged = false;
    for (let s = 0; s < 200 && !converged; s++) {
      const images = band.images();
      const energies = [], gradients = [];
      for (let k = 0; k < images.size(); k++) {
        const [e, g] = surface(images.get(k));
        energies.push(e);
        gradients.push(g);
      }
      band.update(energies, gradients);
      converged = band.step();
    }
    expect(converged).toBe(true);
    expect(band.climbing()).toBe(true);
    const top = band.highestImage();
    expect(bond(band.path().get(top), 0, 1)).toBeCloseTo(2.0, 2);
  });
});
