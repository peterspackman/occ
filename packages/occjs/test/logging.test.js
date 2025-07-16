import { describe, it, expect, beforeAll, beforeEach, afterEach } from 'vitest';
import { loadOCC } from '../src/index.js';

// Helper to extract enum values (Emscripten enums may be objects)
const getEnumValue = (enumVal) => typeof enumVal === 'object' ? enumVal.value : enumVal;

describe('Logging System Tests', () => {
  let Module;
  
  beforeAll(async () => {
    Module = await loadOCC();
  });

  beforeEach(() => {
    // Clear any existing callbacks and settings
    Module.clearLogCallbacks();
    Module.clearLogBuffer();
    Module.setLogBuffering(false);
  });

  afterEach(() => {
    // Clean up after each test
    Module.clearLogCallbacks();
    Module.clearLogBuffer();
    Module.setLogBuffering(false);
  });

  describe('LogLevel Enum', () => {
    it('should have correct enum values', () => {
      expect(getEnumValue(Module.LogLevel.TRACE)).toBe(0);
      expect(getEnumValue(Module.LogLevel.DEBUG)).toBe(1);
      expect(getEnumValue(Module.LogLevel.INFO)).toBe(2);
      expect(getEnumValue(Module.LogLevel.WARN)).toBe(3);
      expect(getEnumValue(Module.LogLevel.ERROR)).toBe(4);
      expect(getEnumValue(Module.LogLevel.CRITICAL)).toBe(5);
      expect(getEnumValue(Module.LogLevel.OFF)).toBe(6);
    });
  });

  describe('Log Level Setting', () => {
    it('should set log level by enum value', () => {
      expect(() => {
        Module.setLogLevel(getEnumValue(Module.LogLevel.DEBUG));
      }).not.toThrow();
    });

    it('should set log level by string', () => {
      expect(() => {
        Module.setLogLevelString("info");
      }).not.toThrow();
    });
  });

  describe('Log Callbacks', () => {
    it('should register callbacks and capture messages', () => {
      const capturedLogs = [];
      
      Module.registerLogCallback((level, message) => {
        capturedLogs.push({ level, message });
      });
      
      Module.setLogLevel(getEnumValue(Module.LogLevel.INFO));
      
      Module.logInfo("Test info message");
      Module.logWarn("Test warning message");
      Module.logError("Test error message");
      
      expect(capturedLogs.length).toBeGreaterThanOrEqual(3);
      
      const infoLog = capturedLogs.find(log => log.message.includes("Test info message"));
      const warnLog = capturedLogs.find(log => log.message.includes("Test warning message"));
      const errorLog = capturedLogs.find(log => log.message.includes("Test error message"));
      
      expect(infoLog).toBeDefined();
      expect(warnLog).toBeDefined();
      expect(errorLog).toBeDefined();
      
      if (infoLog) expect(infoLog.level).toBe(getEnumValue(Module.LogLevel.INFO));
      if (warnLog) expect(warnLog.level).toBe(getEnumValue(Module.LogLevel.WARN));
      if (errorLog) expect(errorLog.level).toBe(getEnumValue(Module.LogLevel.ERROR));
    });

    it('should support multiple callbacks', () => {
      let callback1Count = 0;
      let callback2Count = 0;
      
      Module.registerLogCallback(() => {
        callback1Count++;
      });
      
      Module.registerLogCallback(() => {
        callback2Count++;
      });
      
      Module.logInfo("Test for multiple callbacks");
      
      expect(callback1Count).toBeGreaterThan(0);
      expect(callback2Count).toBeGreaterThan(0);
      expect(callback1Count).toBe(callback2Count);
    });
  });

  describe('Log Buffering', () => {
    it('should buffer log messages when enabled', () => {
      Module.setLogBuffering(true);
      
      Module.logInfo("Buffered info message");
      Module.logWarn("Buffered warning message");
      
      const bufferedLogs = Module.getBufferedLogs();
      
      expect(Array.isArray(bufferedLogs)).toBe(true);
      expect(bufferedLogs.length).toBeGreaterThanOrEqual(2);
      
      if (bufferedLogs.length > 0) {
        const firstLog = bufferedLogs[0];
        expect(firstLog).toHaveProperty('level');
        expect(firstLog).toHaveProperty('message');
        expect(typeof firstLog.level).toBe('number');
        expect(typeof firstLog.message).toBe('string');
      }
      
      const infoLog = bufferedLogs.find(log => log.message.includes("Buffered info message"));
      const warnLog = bufferedLogs.find(log => log.message.includes("Buffered warning message"));
      
      expect(infoLog).toBeDefined();
      expect(warnLog).toBeDefined();
    });

    it('should clear buffer on demand', () => {
      Module.setLogBuffering(true);
      Module.logInfo("Test message");
      
      let logs = Module.getBufferedLogs();
      expect(logs.length).toBeGreaterThan(0);
      
      Module.clearLogBuffer();
      logs = Module.getBufferedLogs();
      expect(logs.length).toBe(0);
    });
  });

  describe('Callback and Buffering Together', () => {
    it('should work simultaneously', () => {
      let callbackCalled = false;
      
      Module.setLogBuffering(true);
      Module.registerLogCallback(() => {
        callbackCalled = true;
      });
      
      Module.logInfo("Test callback with buffering");
      
      expect(callbackCalled).toBe(true);
      
      const bufferedLogs = Module.getBufferedLogs();
      const found = bufferedLogs.some(log => log.message.includes("Test callback with buffering"));
      expect(found).toBe(true);
    });
  });

  describe('Log Level Filtering', () => {
    it('should filter messages based on log level', () => {
      const capturedLogs = [];
      
      Module.registerLogCallback((level, message) => {
        capturedLogs.push({ level, message });
      });
      
      Module.setLogLevel(getEnumValue(Module.LogLevel.WARN));
      
      Module.logDebug("Should not be captured");
      Module.logInfo("Should not be captured");
      Module.logWarn("Should be captured warn");
      Module.logError("Should be captured error");
      
      const debugFound = capturedLogs.some(log => log.message.includes("Should not be captured"));
      const warnFound = capturedLogs.some(log => 
        log.message.includes("Should be captured warn") && 
        log.level === getEnumValue(Module.LogLevel.WARN)
      );
      const errorFound = capturedLogs.some(log => 
        log.message.includes("Should be captured error") && 
        log.level === getEnumValue(Module.LogLevel.ERROR)
      );
      
      expect(debugFound).toBe(false);
      expect(warnFound).toBe(true);
      expect(errorFound).toBe(true);
    });
  });

  describe('All Logging Functions', () => {
    it('should execute all log level functions', () => {
      let messageCount = 0;
      
      Module.registerLogCallback(() => {
        messageCount++;
      });
      
      Module.setLogLevel(getEnumValue(Module.LogLevel.TRACE));
      
      Module.logTrace("Trace message");
      Module.logDebug("Debug message");
      Module.logInfo("Info message");
      Module.logWarn("Warning message");
      Module.logError("Error message");
      Module.logCritical("Critical message");
      
      expect(messageCount).toBeGreaterThanOrEqual(6);
    });
  });

  describe('Log File Setting', () => {
    it('should set log file without throwing', () => {
      expect(() => {
        Module.setLogFile("test.log");
      }).not.toThrow();
    });
  });

  describe('Real-world Usage', () => {
    it('should handle logging during calculations', () => {
      const computationLogs = [];
      
      Module.registerLogCallback((level, message) => {
        computationLogs.push({ level, message });
      });
      
      // Create a simple H2 molecule
      const positions = Module.Mat3N.create(2);
      positions.set(0, 0, 0.0); positions.set(1, 0, 0.0); positions.set(2, 0, 0.0);
      positions.set(0, 1, 0.0); positions.set(1, 1, 0.0); positions.set(2, 1, 0.74);
      
      const atomicNumbers = Module.IVec.fromArray([1, 1]);
      const h2 = new Module.Molecule(atomicNumbers, positions);
      h2.setName("H2");
      
      Module.logInfo(`Created molecule: ${h2.name()} with ${h2.size()} atoms`);
      
      const moleculeLog = computationLogs.find(log => 
        log.message.includes("Created molecule: H2")
      );
      expect(moleculeLog).toBeDefined();
    });
  });
});