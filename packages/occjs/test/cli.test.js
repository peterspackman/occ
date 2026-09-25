import { describe, it, expect, beforeAll } from 'vitest';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// The CLI module (dist/occ.js) behaves like an executable: main() runs on a
// worker thread, Module.runMain resolves with its exit status, and each module
// runs one command.
const distDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '../dist');
let createOccCliModule;

beforeAll(async () => {
  ({ default: createOccCliModule } = await import(path.join(distDir, 'occ.js')));
});

const WATER = `3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
`;

const UREA = `8
urea
O  0.000000  0.000000  1.300000
C  0.000000  0.000000  0.080000
N  0.000000  1.150000 -0.610000
N  0.000000 -1.150000 -0.610000
H  0.000000  2.000000 -0.080000
H  0.000000  1.180000 -1.620000
H  0.000000 -2.000000 -0.080000
H  0.000000 -1.180000 -1.620000
`;

async function freshCli() {
  const output = [];
  const config = {
    noInitialRun: true,
    locateFile: (p) => path.join(distDir, p),
    preRun: [() => { config.ENV.OCC_DATA_PATH = '/'; }],
    print: (line) => output.push(line),
    printErr: (line) => output.push(line),
  };
  const cli = await createOccCliModule(config);
  return { cli, output };
}

function totalEnergy(output) {
  const line = output.find((l) => /^total\s+-?\d/.test(l));
  return line === undefined ? undefined : Number(line.trim().split(/\s+/).pop());
}

describe('occ CLI module', () => {
  it('resolves with status 0 once main has finished, output and files included', async () => {
    const { cli, output } = await freshCli();
    cli.FS.writeFile('/water.xyz', WATER);
    const status = await cli.runMain(['scf', '/water.xyz', 'hf', 'sto-3g']);
    expect(status).toBe(0);
    expect(output.some((l) => /A job well done/.test(l))).toBe(true);
    expect(cli.FS.analyzePath('/water.owf.json').exists).toBe(true);
  });

  it('resolves with a non-zero status when the command fails', async () => {
    const { cli } = await freshCli();
    const status = await cli.runMain(['scf', '/missing.xyz', 'hf', 'sto-3g']);
    expect(status).not.toBe(0);
  });

  it('leaves the host process exit code alone', async () => {
    const { cli } = await freshCli();
    const before = process.exitCode;
    await cli.runMain(['scf', '/missing.xyz', 'hf', 'sto-3g']);
    expect(process.exitCode).toBe(before);
  });

  it('runs one command per module', async () => {
    const { cli } = await freshCli();
    cli.FS.writeFile('/water.xyz', WATER);
    await cli.runMain(['scf', '/water.xyz', 'hf', 'sto-3g']);
    await expect(cli.runMain(['scf', '/water.xyz', 'hf', 'sto-3g'])).rejects.toThrow(/new module/);
  });

  it('refuses callMain, which can no longer report completion', async () => {
    const { cli } = await freshCli();
    expect(() => cli.callMain(['--help'])).toThrow(/runMain/);
  });

  // With main() on the JS thread, 6 or more threads silently ran on one: TBB's
  // workers create further workers, and those creations are serviced only by
  // the JS thread, which was busy running main(). Check the run really is
  // parallel (CPU time well above wall time) and agrees with the serial one.
  it('runs threaded calculations in parallel', async () => {
    const run = async (threads) => {
      const { cli, output } = await freshCli();
      cli.FS.writeFile('/urea.xyz', UREA);
      const cpu0 = process.cpuUsage();
      const wall0 = performance.now();
      const status = await cli.runMain(['scf', '/urea.xyz', 'b3lyp', '6-31g', `--threads=${threads}`]);
      const wall = (performance.now() - wall0) / 1000;
      const cpu = process.cpuUsage(cpu0);
      return { status, energy: totalEnergy(output), cpuPerWall: (cpu.user + cpu.system) / 1e6 / wall };
    };
    const serial = await run(1);
    const threaded = await run(8);
    expect(serial.status).toBe(0);
    expect(threaded.status).toBe(0);
    expect(threaded.energy).toBeCloseTo(serial.energy, 7);
    expect(threaded.cpuPerWall).toBeGreaterThan(1.5);
  }, 180000);
});
