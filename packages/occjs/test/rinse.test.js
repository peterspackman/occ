import { describe, test, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadOCC } from '../dist/index.js';

// The RINSE crystal descriptor, checked against the same reference values as
// the C++ and Python tests (tests/rinse_reference_data.h, from
// rinse-descriptor), and the occ describe JSON records CENext reads.
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, '../../..');
const DATA = path.join(repo, 'tests/data/rinse');
const REFERENCE = path.join(repo, 'tests/rinse_reference_data.h');
const distDir = path.join(here, '../dist');

const CIF_FIXTURES = ['nacl.cif', 'si.cif', 'QEHWEG01_P21.cif', 'QEHWEG03_P21212.cif'];

function reference(name) {
  const text = fs.readFileSync(REFERENCE, 'utf8');
  const start = text.indexOf(`{"${name}",`);
  const end = text.indexOf('\n    {"', start + 1);
  const block = text.slice(start, end === -1 ? text.length : end);
  const hashes = [...block.matchAll(/"([a-z]+(?:-[a-z]+)*)"/g)].map((m) => m[1]);
  const numbers = [...block.matchAll(/-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+/g)].map(
    (m) => Number(m[0]),
  );
  return { oneWord: hashes[0], fiveWords: hashes[1], descriptor: numbers.slice(-128) };
}

const cif = (name) => fs.readFileSync(path.join(DATA, name), 'utf8');

describe('RINSE bindings', () => {
  let occ;

  beforeAll(async () => {
    occ = await loadOCC();
  });

  test('defaults are those of rinse-descriptor', () => {
    const params = new occ.RinseParams();
    expect([params.nMax, params.lMin, params.lMax, params.size()]).toEqual([16, 4, 20, 128]);
    expect(params.qmaxFactor).toBe(1.2);
    expect(params.radialBasis).toBe(occ.RinseRadialBasis.CVGaussian);
    expect(params.lValues()).toEqual([4, 6, 8, 10, 12, 14, 16, 18]);
    expect(params.fixedUiso()).toBeNull();
    params.setFixedUiso(0.02);
    expect(params.fixedUiso()).toBe(0.02);
    params.clearFixedUiso();
    expect(params.fixedUiso()).toBeNull();
    expect(occ.rinseReferenceVersion).toBe('3.2.0');
    params.delete();
  });

  test.each(CIF_FIXTURES)('%s matches the reference', (name) => {
    const expected = reference(name);
    const crystal = occ.Crystal.fromCifString(cif(name));
    const rinse = new occ.Rinse();
    const values = rinse.compute(crystal);
    expect(values.length).toBe(128);
    let worst = 0;
    for (let i = 0; i < 128; i++) worst = Math.max(worst, Math.abs(values[i] - expected.descriptor[i]));
    expect(worst).toBeLessThan(1e-13);
    expect(rinse.hash(crystal, 1)).toBe(expected.oneWord);
    expect(occ.rinseHash(Array.from(values), 5)).toBe(expected.fiveWords);

    const spectrum = rinse.powerSpectrum(crystal);
    expect(spectrum.length).toBe(16);
    expect(spectrum[0].length).toBe(8);
    rinse.delete();
    crystal.delete();
  });
});

describe('occ describe --json in the CLI module', () => {
  test('writes one record per CIF data block', async () => {
    const { default: createOccCliModule } = await import(path.join(distDir, 'occ.js'));
    const output = [];
    const config = {
      noInitialRun: true,
      locateFile: (p) => path.join(distDir, p),
      preRun: [() => { config.ENV.OCC_DATA_PATH = '/'; }],
      print: (line) => output.push(line),
      printErr: (line) => output.push(line),
    };
    const cli = await createOccCliModule(config);
    cli.FS.writeFile(
      '/landscape.cif',
      `${cif('QEHWEG01_P21.cif')}\ndata_notes\n_chemical_name_common 'no cell'\n\n${cif('QEHWEG03_P21212.cif')}\n`,
    );
    const status = await cli.runMain(['describe', '/landscape.cif', '--json', '/rinse.json']);
    expect(status).toBe(0);

    const document = JSON.parse(cli.FS.readFile('/rinse.json', { encoding: 'utf8' }));
    expect(document.descriptor).toBe('rinse');
    expect(document.rinse_version).toBe('3.2.0');
    expect(document.shape).toEqual([16, 8]);
    const { records } = document;
    expect(records.map((r) => r.index)).toEqual([0, 1, 2]);
    expect(records[0].hash).toBe(reference('QEHWEG01_P21.cif').oneWord);
    expect(records[0].values.length).toBe(128);
    expect(records[1].block).toBe('notes');
    expect(records[1].values).toBeNull();
    expect(records[1].error.message).toBe('Missing unit cell data');
    expect(records[2].hash).toBe(reference('QEHWEG03_P21212.cif').oneWord);
  });
});
