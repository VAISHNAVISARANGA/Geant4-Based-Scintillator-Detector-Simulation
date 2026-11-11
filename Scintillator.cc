// =============================================================================
//  CsPbBr3Sim.cc  -  Configurable Scintillator Material and Detector Simulation
//  (single-file Geant4 application; the file name is historical only -
//   NOTHING in here is specific to CsPbBr3. All material data come from config.yaml)
//
//  Layout of this file
//    1. Utilities, units, provenance-aware property parsing
//    2. Configuration loading and validation, stage/mode decision
//    3. Materials, geometry, physics list
//    4. Run data, histograms, JSON/CSV writers
//    5. User actions: primary generator, tracking, stepping, event, run
//    6. main()
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "G4VUserActionInitialization.hh"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4NistManager.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4OpticalParameters.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpticalPhysics.hh"
#include "G4OpticalSurface.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysListFactory.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProcessManager.hh"
#include "G4Run.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VisAttributes.hh"
#include "G4Version.hh"
#include "Randomize.hh"
#ifdef HAVE_G4VIS
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#endif

static const char* SOFTWARE_NAME = "ScintillatorSimulator (CsPbBr3Sim.cc)";
static const char* SOFTWARE_VERSION = "1.0.0";

// ============================================================================
// 1. UTILITIES, UNITS, PROPERTY PARSING
// ============================================================================
struct Issue { char sev; std::string path, msg; };   // 'E' error, 'W' warning, 'I' info
static std::vector<Issue> g_issues;
static void E(const std::string& p, const std::string& m) { g_issues.push_back({'E', p, m}); }
static void W(const std::string& p, const std::string& m) { g_issues.push_back({'W', p, m}); }
static void I(const std::string& p, const std::string& m) { g_issues.push_back({'I', p, m}); }

// Assumptions the program had to make because the user left an optional input out.
static std::vector<std::string> g_assumptionsApplied;

enum class Kind { LEN, TIME, ENERGY, DENS, TEMP, ANGLE, YIELD, FRAC, WL, ELEC, NONE };
static const std::map<std::string, double>& unitTable(Kind k) {
  static const std::map<std::string, double> len{{"mm", 1}, {"cm", 10}, {"m", 1000}, {"um", 1e-3}, {"nm", 1e-6}};
  static const std::map<std::string, double> tm{{"ns", 1}, {"ps", 1e-3}, {"us", 1e3}, {"ms", 1e6}, {"s", 1e9}};
  static const std::map<std::string, double> en{{"eV", 1e-3}, {"keV", 1}, {"MeV", 1e3}, {"GeV", 1e6}};
  static const std::map<std::string, double> de{{"g/cm3", 1}, {"g/cm^3", 1}, {"kg/m3", 1e-3}, {"kg/m^3", 1e-3}};
  static const std::map<std::string, double> te{{"K", 1}};
  static const std::map<std::string, double> an{{"deg", 1}, {"rad", 57.29577951308232}};
  static const std::map<std::string, double> yl{{"photons/MeV", 1}, {"ph/MeV", 1}, {"photons/keV", 1e3}, {"ph/keV", 1e3}};
  static const std::map<std::string, double> fr{{"", 1}, {"1", 1}, {"fraction", 1}, {"%", 0.01}};
  static const std::map<std::string, double> wl{{"nm", 1}, {"um", 1e3}, {"angstrom", 0.1}};
  static const std::map<std::string, double> el{{"electrons", 1}, {"e", 1}};
  static const std::map<std::string, double> no{{"", 1}, {"1", 1}};
  switch (k) {
    case Kind::LEN: return len; case Kind::TIME: return tm; case Kind::ENERGY: return en;
    case Kind::DENS: return de; case Kind::TEMP: return te; case Kind::ANGLE: return an;
    case Kind::YIELD: return yl; case Kind::FRAC: return fr; case Kind::WL: return wl;
    case Kind::ELEC: return el;
    default: return no;
  }
}
static bool kindUnitOptional(Kind k) { return k == Kind::FRAC || k == Kind::NONE; }

// One record per scientific input; written to summary.json ("provenance").
struct ProvRec { std::string path, valueText, unit, status, source, conditions; bool known; };
static std::vector<ProvRec> g_prov;

struct Prop {
  bool present = false, known = false, infinite = false, hasTable = false;
  double value = 0;                 // converted to base units (mm, ns, keV, g/cm3, K, deg, photons/MeV, fraction)
  std::string path, unit, status, source, conditions, valueText;
  std::vector<double> tx, ty;       // optional table: x = wavelength [nm] ascending, y in base units
};

static const std::set<std::string> kStatuses{"experimental", "literature", "calculated", "estimated", "assumed", "unknown"};

// Safe child access: never throws on missing/zombie/non-map nodes (returns a null node instead).
static YAML::Node sub(const YAML::Node& n, const std::string& k) {
  if (!n.IsDefined() || !n.IsMap()) return YAML::Node();
  const YAML::Node r = n[k];
  if (!r.IsDefined()) return YAML::Node();
  return r;
}
// NOTE: never assign to a YAML::Node that aliases the document (yaml-cpp would overwrite it); recurse instead.
static YAML::Node atRec(const YAML::Node& n, const std::vector<std::string>& p, size_t i) {
  if (i == p.size()) return n;
  YAML::Node c = sub(n, p[i]);
  if (!c.IsDefined() || c.IsNull()) return YAML::Node();
  return atRec(c, p, i + 1);
}
static YAML::Node at(const YAML::Node& root, const std::string& dotted) {
  std::vector<std::string> parts; std::stringstream ss(dotted); std::string part;
  while (std::getline(ss, part, '.')) parts.push_back(part);
  return atRec(root, parts, 0);
}
static std::string strOf(const YAML::Node& n, const std::string& dflt = "") {
  if (!n || !n.IsDefined() || n.IsNull() || !n.IsScalar()) return dflt;
  return n.Scalar();
}
static bool parseDouble(const std::string& s, double& v) {
  char* e = nullptr;
  v = std::strtod(s.c_str(), &e);
  return e != s.c_str() && *e == '\0' && std::isfinite(v);
}
static bool isInfText(const std::string& s) {
  return s == "inf" || s == ".inf" || s == "infinite" || s == "Infinity" || s == ".Inf";
}

// Parse a provenance-carrying property. Never inserts defaults.
static Prop parseProp(const YAML::Node& n, const std::string& path, Kind kind, bool allowInf = false,
                      bool allowTable = false) {
  Prop p; p.path = path;
  if (!n || !n.IsDefined() || n.IsNull()) return p;           // not present
  if (!n.IsMap()) { E(path, "must be a mapping with value/unit/status/source"); return p; }
  p.present = true;
  p.status = strOf(sub(n,"status"));
  p.source = strOf(sub(n,"source"));
  p.conditions = strOf(sub(n,"conditions"));
  p.unit = strOf(sub(n,"unit"));
  if (p.status.empty()) E(path, "missing 'status' (experimental|literature|calculated|estimated|assumed|unknown)");
  else if (!kStatuses.count(p.status)) E(path, "invalid status '" + p.status + "'");
  auto record = [&]() { g_prov.push_back({path, p.valueText, p.unit, p.status, p.source, p.conditions, p.known}); };
  if (p.status == "unknown") { p.valueText = "unknown"; record(); return p; }
  if ((p.status == "literature" || p.status == "experimental") && p.source.empty())
    W(path, "status '" + p.status + "' but no 'source' given");

  const YAML::Node vn = sub(n,"value");
  bool hasVal = vn && vn.IsDefined() && !vn.IsNull();
  // a table may be given nested under "table:" or directly (x_unit/points next to status/source)
  const YAML::Node tn0 = sub(n, "table");
  const YAML::Node tn = (tn0.IsDefined() && !tn0.IsNull()) ? tn0 : n;
  bool hasTab = allowTable && tn.IsMap() && sub(tn, "points").IsDefined() && sub(tn, "points").IsSequence() && sub(tn, "points").size() > 0;
  if (!hasVal && !hasTab) { p.valueText = "not provided"; record(); return p; }

  const auto& ut = unitTable(kind);
  double f = 1.0;
  if (!(kindUnitOptional(kind) && p.unit.empty())) {
    auto it = ut.find(p.unit);
    if (it == ut.end()) {
      std::string acc; for (auto& kv : ut) acc += (acc.empty() ? "" : ", ") + (kv.first.empty() ? "\"\"" : kv.first);
      E(path, "unit '" + p.unit + "' not accepted here; accepted: " + acc);
      record(); return p;
    }
    f = it->second;
  }
  if (hasVal) {
    std::string s = strOf(vn);
    if (isInfText(s)) {
      if (!allowInf) { E(path, "infinite value not allowed here"); record(); return p; }
      p.infinite = true; p.value = 1e30; p.valueText = "inf";
    } else {
      double v;
      if (!parseDouble(s, v)) { E(path, "value '" + s + "' is not a finite number"); record(); return p; }
      p.value = v * f; p.valueText = s;
    }
  }
  if (hasTab) {
    std::string xu = strOf(sub(tn,"x_unit"));
    double xf = 1.0; bool xEnergy = false;
    if (xu == "nm") xf = 1; else if (xu == "um") xf = 1e3; else if (xu == "angstrom") xf = 0.1;
    else if (xu == "eV") xEnergy = true;
    else { E(path + ".table.x_unit", "must be nm, um, angstrom or eV"); record(); return p; }
    std::vector<std::pair<double, double>> pts;
    for (const auto& pt : sub(tn,"points")) {
      double x, y;
      if (!pt.IsSequence() || pt.size() != 2 || !parseDouble(strOf(pt[0]), x) || !parseDouble(strOf(pt[1]), y)) {
        E(path + ".table.points", "each point must be [x, y] with finite numbers"); record(); return p;
      }
      if (xEnergy) { if (x <= 0) { E(path + ".table.points", "photon energy must be > 0"); record(); return p; } x = 1239.84198 / x; }
      else x *= xf;
      pts.push_back({x, y * (p.infinite ? 1 : f)});
    }
    std::sort(pts.begin(), pts.end());
    if (pts.size() < 2) { E(path + ".table.points", "need at least 2 points"); record(); return p; }
    for (size_t i = 1; i < pts.size(); ++i)
      if (!(pts[i].first > pts[i - 1].first)) { E(path + ".table.points", "duplicate x values"); record(); return p; }
    for (auto& q : pts) { p.tx.push_back(q.first); p.ty.push_back(q.second); }
    p.hasTable = true;
    if (!hasVal) p.valueText = "table(" + std::to_string(pts.size()) + " points)";
    else p.valueText += " (table overrides)";
  }
  p.known = true;
  record();
  return p;
}

// ============================================================================
// 2. CONFIGURATION
// ============================================================================
struct Comp { Prop frac, tau; };
struct Config {
  // simulation / output
  std::string modeRequested, modeEffective, physList, outDir, photonLog, visMacro;
  double cutMM = 0; long events = 0, seed = 0, maxPhotonLog = 0;
  // material
  std::string matName, state;
  std::vector<std::pair<std::string, double>> elems;
  Prop density, temperature, stateProp, compProv;
  // crystal + surface + surroundings
  std::string shape, surfModel, finish, reflType, surrMat;
  double dx = 0, dy = 0, dz = 0;
  Prop sigmaAlpha, roughness, reflectivity, surrN, lobeSpec, lobeSpike, lobeBack;
  // source
  std::string srcType, particle;
  double eGammaKeV = 0, coneDeg = 180;
  G4ThreeVector srcPos, srcDir;
  // scintillation
  Prop lightYield, resScale, peakWL, fwhm, riseTime, specTable;
  std::string specMode, specBasis;
  std::vector<Comp> comps;
  // optical
  Prop nCrystal, absLen, rayLen, mieLen, mieFwd, mieBwd, mieRatio, wlsAbs, wlsMean, wlsTau, wlsEm;
  bool rayEnabled = false, mieEnabled = false, wlsEnabled = false;
  std::string wlsBasis;
  // sensor
  std::string sType, sFace, winMat, coupType, coupMat;
  double sW = 0, sH = 0, sOffU = 0, sOffV = 0, sThick = 0;
  Prop winN, pde, coupT, coupN, cells, gain, recov, pmtGain, transit, jitter, noise;
  bool sipmSat = true, sipmRec = true, pmtEnabled = false;
  std::vector<std::pair<std::string, std::string>> cliOverrides;
};
static Config C;
static YAML::Node g_root;
static std::string g_configText, g_configPath;

// Stage bookkeeping ----------------------------------------------------------
struct StageInfo { bool ok = true; std::vector<std::string> missing; };
static std::map<std::string, StageInfo> g_stage;   // gamma_screening, scintillation, optical_transport, photodetection
static void need(const std::string& stage, const Prop& p, const std::string& label) {
  if (!p.known) {
    g_stage[stage].ok = false;
    g_stage[stage].missing.push_back(label + (p.present ? (p.status == "unknown" ? " (status: unknown)" : " (status: " + p.status + ", " + p.valueText + ")") : " (not provided)"));
  }
}
static void needCond(const std::string& stage, bool cond, const std::string& what) {
  if (!cond) { g_stage[stage].ok = false; g_stage[stage].missing.push_back(what); }
}

static std::string oneOf(const YAML::Node& root, const std::string& path, const std::set<std::string>& allowed,
                         bool required = true) {
  std::string s = strOf(at(root, path));
  if (s.empty()) { if (required) E(path, "required"); return ""; }
  if (!allowed.count(s)) {
    std::string acc; for (auto& a : allowed) acc += (acc.empty() ? "" : ", ") + a;
    E(path, "'" + s + "' not supported; allowed: " + acc);
  }
  return s;
}
static bool getNum(const YAML::Node& n, const std::string& path, double& v, bool required = true) {
  std::string s = strOf(n);
  if (s.empty()) { if (required) E(path, "required number missing"); return false; }
  if (!parseDouble(s, v)) { E(path, "'" + s + "' is not a finite number"); return false; }
  return true;
}
static void range(const Prop& p, double lo, double hi, const std::string& what, bool loInclusive = true) {
  if (!p.known || p.infinite) return;
  bool bad = (loInclusive ? p.value < lo : p.value <= lo) || p.value > hi;
  if (bad) E(p.path, what + " out of physical range [" + std::to_string(lo) + ", " + std::to_string(hi) + "] (after unit conversion: " + std::to_string(p.value) + ")");
  for (double y : p.ty) if (!p.infinite && ((loInclusive ? y < lo : y <= lo) || y > hi)) { E(p.path + ".table", what + " table value out of range"); break; }
}

// Load + validate the whole YAML file. Returns false if there are errors.
static bool LoadConfig(const std::string& file) {
  g_configPath = file;
  { std::ifstream f(file); if (!f) { E(file, "cannot open configuration file"); return false; }
    std::stringstream ss; ss << f.rdbuf(); g_configText = ss.str(); }
  try { g_root = YAML::Load(g_configText); }
  catch (const std::exception& ex) { E(file, std::string("YAML syntax error: ") + ex.what()); return false; }
  const YAML::Node& R = g_root;
  if (!R.IsMap()) { E(file, "top level must be a mapping"); return false; }
  static const std::set<std::string> topKeys{"config_version", "simulation", "output", "visualization", "material", "crystal",
                                              "surface", "surrounding", "source", "scintillation", "optical", "sensor"};
  for (auto it = R.begin(); it != R.end(); ++it)
    if (!topKeys.count(it->first.as<std::string>())) W(it->first.as<std::string>(), "unknown top-level section (ignored)");
  if (strOf(sub(R,"config_version")) != "1") W("config_version", "expected 1");

  g_stage["gamma_screening"]; g_stage["scintillation"]; g_stage["optical_transport"]; g_stage["photodetection"];

  // ---- simulation / output ------------------------------------------------
  C.modeRequested = oneOf(R, "simulation.mode", {"auto", "screening", "full"});
  C.physList = strOf(at(R, "simulation.physics_list"));
  if (C.physList.empty()) E("simulation.physics_list", "required");
  {
    const YAML::Node cn = at(R, "simulation.production_cut");
    double v; std::string u = strOf(sub(cn,"unit"));
    auto it = unitTable(Kind::LEN).find(u);
    if (!cn || !getNum(sub(cn,"value"), "simulation.production_cut.value", v)) {}
    else if (it == unitTable(Kind::LEN).end()) E("simulation.production_cut.unit", "unit must be a length (mm, cm, um, ...)");
    else { C.cutMM = v * it->second; if (C.cutMM <= 0 || C.cutMM > 100) E("simulation.production_cut", "must be in (0,100] mm"); }
  }
  { double v; if (getNum(at(R, "simulation.events"), "simulation.events", v)) { C.events = (long)v; if (v < 1 || v > 1e8 || v != std::floor(v)) E("simulation.events", "must be an integer in [1, 1e8]"); } }
  { double v; if (getNum(at(R, "simulation.seed"), "simulation.seed", v)) { C.seed = (long)v; if (v < 0 || v != std::floor(v)) E("simulation.seed", "must be a non-negative integer"); } }
  C.outDir = strOf(at(R, "output.directory"), "");
  if (C.outDir.empty()) E("output.directory", "required");
  C.photonLog = oneOf(R, "output.photon_log", {"all", "sensor", "none"});
  { double v = 2e6; getNum(at(R, "output.max_logged_photons"), "output.max_logged_photons", v, false); C.maxPhotonLog = (long)v; }
  C.visMacro = strOf(at(R, "visualization.macro"), "vis.mac");

  // ---- A. material -----------------------------------------------------------
  C.matName = strOf(at(R, "material.name"));
  if (C.matName.empty()) E("material.name", "required");
  C.state = strOf(at(R, "material.physical_state.value"));
  if (C.state != "solid" && C.state != "liquid" && C.state != "gas") E("material.physical_state.value", "must be solid, liquid or gas");
  { const YAML::Node ps = at(R, "material.physical_state");
    std::string st = strOf(sub(ps, "status"));
    if (!kStatuses.count(st)) E("material.physical_state.status", "required: experimental|literature|calculated|estimated|assumed|unknown");
    g_prov.push_back({"material.physical_state", C.state, "", st, strOf(sub(ps, "source")), strOf(sub(ps, "conditions")), !C.state.empty()}); }
  {
    const YAML::Node cn = at(R, "material.composition");
    C.compProv.path = "material.composition";
    if (!cn || !cn.IsMap()) E("material.composition", "required mapping with 'elements', 'status', 'source'");
    else {
      std::string st = strOf(sub(cn,"status"));
      if (!kStatuses.count(st)) E("material.composition.status", "required: experimental|literature|calculated|estimated|assumed|unknown");
      C.compProv.status = st; C.compProv.source = strOf(sub(cn,"source")); C.compProv.conditions = strOf(sub(cn,"conditions"));
      std::map<std::string, double> merged; std::vector<std::string> order;
      const YAML::Node en = sub(cn,"elements");
      if (!en || !en.IsSequence() || en.size() == 0) E("material.composition.elements", "need a non-empty list of {symbol, atoms}");
      else {
        auto* nm = G4NistManager::Instance();
        std::string txt;
        for (const auto& e : en) {
          std::string sym = strOf(sub(e,"symbol")); double a;
          if (sym.empty() || !getNum(sub(e,"atoms"), "material.composition.elements[" + sym + "].atoms", a)) { E("material.composition.elements", "each entry needs 'symbol' and 'atoms'"); continue; }
          if (a <= 0) { E("material.composition.elements", "atoms for " + sym + " must be > 0"); continue; }
          if (nm->GetZ(sym) <= 0) { E("material.composition.elements", "'" + sym + "' is not a valid element symbol (case sensitive, e.g. Cs, Pb, Br)"); continue; }
          if (!merged.count(sym)) order.push_back(sym);
          merged[sym] += a;
          txt += sym + std::to_string(a) + " ";
        }
        for (auto& s : order) C.elems.push_back({s, merged[s]});
        g_prov.push_back({"material.composition", txt, "atoms per formula unit", C.compProv.status, C.compProv.source, C.compProv.conditions, !C.elems.empty()});
        if (C.compProv.status == "unknown") { g_stage["gamma_screening"].ok = false; g_stage["gamma_screening"].missing.push_back("material.composition (status unknown)"); }
      }
    }
  }
  C.density = parseProp(at(R, "material.density"), "material.density", Kind::DENS);
  need("gamma_screening", C.density, "material.density");
  range(C.density, 1e-3, 25.0, "density [g/cm3]", false);
  C.temperature = parseProp(at(R, "material.temperature"), "material.temperature", Kind::TEMP);
  need("gamma_screening", C.temperature, "material.temperature");
  range(C.temperature, 0.0, 3000.0, "temperature [K]", false);
  if (C.temperature.known && C.temperature.value > 1000) W(C.temperature.path, "temperature above 1000 K is unusual for a scintillator");
  if (C.compProv.status == "estimated" || C.density.status == "estimated") I("material", "composition/density marked 'estimated' - results are conditional on those estimates");

  // ---- B. geometry / surface ------------------------------------------------------
  C.shape = oneOf(R, "crystal.shape", {"box"});
  {
    const YAML::Node d = at(R, "crystal.dimensions");
    std::string u = strOf(sub(d,"unit"));
    auto it = unitTable(Kind::LEN).find(u);
    if (!d || it == unitTable(Kind::LEN).end()) E("crystal.dimensions.unit", "required length unit (mm, cm, ...)");
    else {
      double x, y, z;
      bool ok = getNum(sub(d,"x"), "crystal.dimensions.x", x) & getNum(sub(d,"y"), "crystal.dimensions.y", y) & getNum(sub(d,"z"), "crystal.dimensions.z", z);
      if (ok) {
        C.dx = x * it->second; C.dy = y * it->second; C.dz = z * it->second;
        for (double v : {C.dx, C.dy, C.dz}) if (v < 0.05 || v > 1000) E("crystal.dimensions", "each dimension must be within [0.05, 1000] mm");
      }
    }
  }
  C.surfModel = oneOf(R, "surface.model", {"unified"});
  C.finish = oneOf(R, "surface.finish", {"polished", "ground"});
  C.sigmaAlpha = parseProp(at(R, "surface.sigma_alpha"), "surface.sigma_alpha", Kind::ANGLE);
  C.roughness = parseProp(at(R, "surface.roughness_rms"), "surface.roughness_rms", Kind::LEN);
  C.reflType = oneOf(R, "surface.reflector.type", {"none", "specular", "diffuse"});
  C.reflectivity = parseProp(at(R, "surface.reflector.reflectivity"), "surface.reflector.reflectivity", Kind::FRAC);
  range(C.reflectivity, 0.0, 1.0, "reflectivity");
  if (C.reflType != "none") I("surface.finish", "with a reflector, reflector.type sets the Geant4 finish (specular: polishedfrontpainted, diffuse: groundfrontpainted); 'finish' is used only when reflector.type is none");
  {
    const YAML::Node uc = at(R, "surface.unified_constants");
    C.lobeSpec  = parseProp(sub(uc,"specular_lobe"),  "surface.unified_constants.specular_lobe",  Kind::FRAC);
    C.lobeSpike = parseProp(sub(uc,"specular_spike"), "surface.unified_constants.specular_spike", Kind::FRAC);
    C.lobeBack  = parseProp(sub(uc,"backscatter"),    "surface.unified_constants.backscatter",    Kind::FRAC);
    for (auto* p : {&C.lobeSpec, &C.lobeSpike, &C.lobeBack}) range(*p, 0, 1, "unified constant");
  }
  C.surrMat = strOf(at(R, "surrounding.nist_material"));
  if (C.surrMat.empty()) E("surrounding.nist_material", "required");
  else if (!G4NistManager::Instance()->FindOrBuildMaterial(C.surrMat, false)) E("surrounding.nist_material", "unknown Geant4 NIST material '" + C.surrMat + "'");
  C.surrN = parseProp(at(R, "surrounding.refractive_index"), "surrounding.refractive_index", Kind::NONE, false, true);
  range(C.surrN, 1.0, 5.0, "refractive index");

  // ---- C. source ---------------------------------------------------------------
  C.particle = oneOf(R, "source.particle", {"gamma"});
  C.srcType = oneOf(R, "source.type", {"pencil_beam", "point_source", "isotropic_source"});
  {
    const YAML::Node en = at(R, "source.energy");
    double v; std::string u = strOf(sub(en,"unit"));
    auto it = unitTable(Kind::ENERGY).find(u);
    if (!en || !getNum(sub(en,"value"), "source.energy.value", v)) {}
    else if (it == unitTable(Kind::ENERGY).end()) E("source.energy.unit", "must be eV, keV, MeV or GeV");
    else { C.eGammaKeV = v * it->second;
      if (C.eGammaKeV < 1.0 || C.eGammaKeV > 1e5) E("source.energy", "gamma energy must be within [1 keV, 100 MeV]");
      else if (C.eGammaKeV > 1e4) W("source.energy", "above 10 MeV: scintillation light yield is usually not measured there; interpret with care"); }
    const YAML::Node pn = at(R, "source.position");
    std::string pu = strOf(sub(pn,"unit")); auto pit = unitTable(Kind::LEN).find(pu);
    double x, y, z;
    if (!pn || pit == unitTable(Kind::LEN).end()) E("source.position.unit", "required length unit");
    else if (getNum(sub(pn,"x"), "source.position.x", x) & getNum(sub(pn,"y"), "source.position.y", y) & getNum(sub(pn,"z"), "source.position.z", z))
      C.srcPos = G4ThreeVector(x, y, z) * pit->second;
    const YAML::Node dn = at(R, "source.direction");
    if (!dn) E("source.direction", "required");
    else if (getNum(sub(dn,"x"), "source.direction.x", x) & getNum(sub(dn,"y"), "source.direction.y", y) & getNum(sub(dn,"z"), "source.direction.z", z)) {
      G4ThreeVector d(x, y, z);
      if (d.mag() < 1e-9) E("source.direction", "must be a non-zero vector"); else C.srcDir = d.unit();
    }
    if (C.srcType == "point_source") {
      const YAML::Node ca = at(R, "source.cone_half_angle");
      double cv; auto cit = unitTable(Kind::ANGLE).find(strOf(sub(ca, "unit")));
      if (!ca || cit == unitTable(Kind::ANGLE).end() || !getNum(sub(ca, "value"), "source.cone_half_angle.value", cv)) E("source.cone_half_angle", "required for point_source: {value, unit: deg|rad}");
      else { C.coneDeg = cv * cit->second; if (C.coneDeg <= 0 || C.coneDeg > 180) E("source.cone_half_angle", "must be in (0, 180] deg"); }
    } else if (C.srcType == "isotropic_source") C.coneDeg = 180;
    else { C.coneDeg = 0; if (!at(R, "source.cone_half_angle").IsNull()) I("source.cone_half_angle", "ignored for pencil_beam (used only by point_source)"); }
  }

  // ---- D. scintillation ---------------------------------------------------------------
  C.lightYield = parseProp(at(R, "scintillation.light_yield"), "scintillation.light_yield", Kind::YIELD);
  need("scintillation", C.lightYield, "scintillation.light_yield");
  range(C.lightYield, 0.0, 1e6, "light yield [photons/MeV]", false);
  if (C.lightYield.known && C.lightYield.value > 1.5e5) W(C.lightYield.path, "above 150,000 photons/MeV exceeds known scintillators - check the value");
  C.resScale = parseProp(at(R, "scintillation.resolution_scale"), "scintillation.resolution_scale", Kind::NONE);
  need("scintillation", C.resScale, "scintillation.resolution_scale (1 = Poisson; state explicitly)");
  range(C.resScale, 0.0, 100.0, "resolution scale");
  C.peakWL = parseProp(at(R, "scintillation.emission.peak_wavelength"), "scintillation.emission.peak_wavelength", Kind::WL);
  need("scintillation", C.peakWL, "scintillation.emission.peak_wavelength");
  range(C.peakWL, 150, 2000, "emission peak [nm]", false);
  C.fwhm = parseProp(at(R, "scintillation.emission.fwhm"), "scintillation.emission.fwhm", Kind::WL);
  range(C.fwhm, 0.0, 1000, "emission FWHM [nm]", false);
  C.specMode = oneOf(R, "scintillation.emission.spectrum_mode", {"gaussian", "table", "monochromatic"});
  C.specTable = parseProp(at(R, "scintillation.emission.spectrum_table"), "scintillation.emission.spectrum_table", Kind::NONE, false, true);
  C.specBasis = strOf(at(R, "scintillation.emission.spectrum_table.intensity_basis"), "per_wavelength");
  if (C.specMode == "gaussian") { need("scintillation", C.fwhm, "scintillation.emission.fwhm (required for spectrum_mode: gaussian)"); }
  if (C.specMode == "table") {
    needCond("scintillation", C.specTable.known && C.specTable.hasTable, "scintillation.emission.spectrum_table (required for spectrum_mode: table)");
    for (double y : C.specTable.ty) if (y < 0) { E(C.specTable.path, "intensities must be >= 0"); break; }
  }
  C.riseTime = parseProp(at(R, "scintillation.rise_time"), "scintillation.rise_time", Kind::TIME);
  range(C.riseTime, 0.0, 1e5, "rise time [ns]");
  {
    const YAML::Node dc = at(R, "scintillation.decay_components");
    if (!dc || !dc.IsSequence() || dc.size() == 0) { g_stage["scintillation"].ok = false; g_stage["scintillation"].missing.push_back("scintillation.decay_components (need at least one {fraction, decay_time})"); }
    else if (dc.size() > 3) E("scintillation.decay_components", "Geant4 supports at most 3 scintillation components");
    else {
      int i = 0; double sum = 0; bool allKnown = true;
      for (const auto& c : dc) {
        ++i; Comp cp;
        cp.frac = parseProp(sub(c,"fraction"), "scintillation.decay_components[" + std::to_string(i) + "].fraction", Kind::FRAC);
        cp.tau = parseProp(sub(c,"decay_time"), "scintillation.decay_components[" + std::to_string(i) + "].decay_time", Kind::TIME);
        need("scintillation", cp.frac, cp.frac.path); need("scintillation", cp.tau, cp.tau.path);
        range(cp.frac, 0.0, 1.0, "fraction", false); range(cp.tau, 0.0, 1e7, "decay time [ns]", false);
        if (cp.frac.known) sum += cp.frac.value; else allKnown = false;
        if (!cp.tau.known) allKnown = false;
        C.comps.push_back(cp);
      }
      if (allKnown && std::fabs(sum - 1.0) > 1e-6) E("scintillation.decay_components", "fractions must sum to 1 (sum = " + std::to_string(sum) + ")");
    }
  }

  // ---- E. optical ------------------------------------------------------------------------
  C.nCrystal = parseProp(at(R, "optical.refractive_index"), "optical.refractive_index", Kind::NONE, false, true);
  need("optical_transport", C.nCrystal, "optical.refractive_index");
  range(C.nCrystal, 1.0, 6.0, "refractive index");
  C.absLen = parseProp(at(R, "optical.absorption_length"), "optical.absorption_length", Kind::LEN, true, true);
  need("optical_transport", C.absLen, "optical.absorption_length (use value: inf with status estimated for a transparent bulk)");
  range(C.absLen, 1e-4, 1e9, "absorption length [mm]", false);
  need("optical_transport", C.surrN, "surrounding.refractive_index");
  C.rayEnabled = strOf(at(R, "optical.rayleigh.enabled")) == "true";
  C.mieEnabled = strOf(at(R, "optical.mie.enabled")) == "true";
  C.wlsEnabled = strOf(at(R, "optical.wls.enabled")) == "true";
  C.rayLen = parseProp(at(R, "optical.rayleigh.scattering_length"), "optical.rayleigh.scattering_length", Kind::LEN, true, true);
  if (C.rayEnabled) { need("optical_transport", C.rayLen, "optical.rayleigh.scattering_length (rayleigh.enabled: true)"); range(C.rayLen, 1e-4, 1e9, "Rayleigh length [mm]", false); }
  C.mieLen = parseProp(at(R, "optical.mie.scattering_length"), "optical.mie.scattering_length", Kind::LEN, true, true);
  C.mieFwd = parseProp(at(R, "optical.mie.forward_anisotropy"), "optical.mie.forward_anisotropy", Kind::NONE);
  C.mieBwd = parseProp(at(R, "optical.mie.backward_anisotropy"), "optical.mie.backward_anisotropy", Kind::NONE);
  C.mieRatio = parseProp(at(R, "optical.mie.forward_ratio"), "optical.mie.forward_ratio", Kind::FRAC);
  if (C.mieEnabled) {
    need("optical_transport", C.mieLen, "optical.mie.scattering_length (mie.enabled: true)");
    need("optical_transport", C.mieFwd, "optical.mie.forward_anisotropy"); need("optical_transport", C.mieBwd, "optical.mie.backward_anisotropy");
    need("optical_transport", C.mieRatio, "optical.mie.forward_ratio");
    range(C.mieLen, 1e-4, 1e9, "Mie length [mm]", false); range(C.mieFwd, -1, 1, "g_forward"); range(C.mieBwd, -1, 1, "g_backward"); range(C.mieRatio, 0, 1, "forward ratio");
  }
  C.wlsAbs = parseProp(at(R, "optical.wls.absorption_length"), "optical.wls.absorption_length", Kind::LEN, false, true);
  C.wlsMean = parseProp(at(R, "optical.wls.mean_photons"), "optical.wls.mean_photons", Kind::NONE);
  C.wlsTau = parseProp(at(R, "optical.wls.time_constant"), "optical.wls.time_constant", Kind::TIME);
  C.wlsEm = parseProp(at(R, "optical.wls.emission_table"), "optical.wls.emission_table", Kind::NONE, false, true);
  C.wlsBasis = strOf(at(R, "optical.wls.emission_table.intensity_basis"), "per_wavelength");
  if (C.wlsEnabled) {
    need("optical_transport", C.wlsAbs, "optical.wls.absorption_length (wls.enabled: true)");
    need("optical_transport", C.wlsMean, "optical.wls.mean_photons"); need("optical_transport", C.wlsTau, "optical.wls.time_constant");
    needCond("optical_transport", C.wlsEm.known && C.wlsEm.hasTable, "optical.wls.emission_table (wls.enabled: true)");
    range(C.wlsMean, 0.0, 100, "WLS mean photons", false); range(C.wlsTau, 0.0, 1e6, "WLS time constant [ns]");
  }
  { std::string sm = strOf(at(R, "optical.rayleigh.enabled")); if (!sm.empty() && sm != "true" && sm != "false") E("optical.rayleigh.enabled", "must be true or false"); }
  { std::string sm = strOf(at(R, "optical.mie.enabled")); if (!sm.empty() && sm != "true" && sm != "false") E("optical.mie.enabled", "must be true or false"); }
  { std::string sm = strOf(at(R, "optical.wls.enabled")); if (!sm.empty() && sm != "true" && sm != "false") E("optical.wls.enabled", "must be true or false"); }

  // surface physics prerequisites
  if (C.reflType == "none") {
    if (C.finish == "ground") need("optical_transport", C.sigmaAlpha, "surface.sigma_alpha (finish: ground)");
  } else {
    need("optical_transport", C.reflectivity, "surface.reflector.reflectivity (reflector.type: " + C.reflType + ")");
  }
  if (C.finish == "ground" && C.reflType == "none") {
    range(C.sigmaAlpha, 0, 90, "sigma_alpha [deg]");
  }

  // ---- F. sensor ---------------------------------------------------------------------------
  C.sType = oneOf(R, "sensor.type", {"SiPM", "PMT"});
  C.sFace = oneOf(R, "sensor.face", {"+x", "-x", "+y", "-y", "+z", "-z"});
  {
    const YAML::Node a = at(R, "sensor.active_area");
    std::string u = strOf(sub(a,"unit")); auto it = unitTable(Kind::LEN).find(u);
    if (!a || it == unitTable(Kind::LEN).end()) { needCond("photodetection", false, "sensor.active_area (width, height, unit)"); }
    else {
      double w, h, ou = 0, ov = 0;
      bool ok = getNum(sub(a,"width"), "sensor.active_area.width", w) & getNum(sub(a,"height"), "sensor.active_area.height", h);
      getNum(sub(a,"offset_u"), "sensor.active_area.offset_u", ou, false); getNum(sub(a,"offset_v"), "sensor.active_area.offset_v", ov, false);
      if (ok) { C.sW = w * it->second; C.sH = h * it->second; C.sOffU = ou * it->second; C.sOffV = ov * it->second;
        if (C.sW <= 0 || C.sH <= 0) E("sensor.active_area", "width/height must be > 0"); }
      else needCond("photodetection", false, "sensor.active_area");
    }
    const YAML::Node t = at(R, "sensor.model_thickness");
    double v; std::string tu = strOf(sub(t,"unit")); auto tit = unitTable(Kind::LEN).find(tu);
    if (t && tit != unitTable(Kind::LEN).end() && getNum(sub(t,"value"), "sensor.model_thickness.value", v)) { C.sThick = v * tit->second; if (C.sThick <= 0 || C.sThick > 50) E("sensor.model_thickness", "must be in (0, 50] mm"); }
    else E("sensor.model_thickness", "required {value, unit}");
  }
  C.winMat = strOf(at(R, "sensor.window.nist_material"));
  if (C.winMat.empty()) E("sensor.window.nist_material", "required");
  else if (!G4NistManager::Instance()->FindOrBuildMaterial(C.winMat, false)) E("sensor.window.nist_material", "unknown NIST material '" + C.winMat + "'");
  C.winN = parseProp(at(R, "sensor.window.refractive_index"), "sensor.window.refractive_index", Kind::NONE, false, true);
  need("photodetection", C.winN, "sensor.window.refractive_index"); range(C.winN, 1.0, 6.0, "refractive index");
  C.coupType = oneOf(R, "sensor.coupling.type", {"direct", "layer"});
  C.coupMat = strOf(at(R, "sensor.coupling.nist_material"));
  C.coupT = parseProp(at(R, "sensor.coupling.thickness"), "sensor.coupling.thickness", Kind::LEN);
  C.coupN = parseProp(at(R, "sensor.coupling.refractive_index"), "sensor.coupling.refractive_index", Kind::NONE, false, true);
  if (C.coupType == "layer") {
    if (C.coupMat.empty()) E("sensor.coupling.nist_material", "required for coupling.type: layer");
    else if (!G4NistManager::Instance()->FindOrBuildMaterial(C.coupMat, false)) E("sensor.coupling.nist_material", "unknown NIST material '" + C.coupMat + "'");
    need("photodetection", C.coupT, "sensor.coupling.thickness"); need("photodetection", C.coupN, "sensor.coupling.refractive_index");
    range(C.coupT, 1e-3, 20, "coupling thickness [mm]", false); range(C.coupN, 1.0, 6.0, "refractive index");
  }
  C.pde = parseProp(at(R, "sensor.pde"), "sensor.pde", Kind::FRAC, false, true);
  need("photodetection", C.pde, "sensor.pde"); range(C.pde, 0.0, 1.0, "PDE");
  C.sipmSat = strOf(at(R, "sensor.sipm.model_saturation"), "true") == "true";
  C.sipmRec = strOf(at(R, "sensor.sipm.model_recovery"), "true") == "true";
  C.cells = parseProp(at(R, "sensor.sipm.microcell_count"), "sensor.sipm.microcell_count", Kind::NONE);
  C.gain = parseProp(at(R, "sensor.sipm.gain"), "sensor.sipm.gain", Kind::NONE);
  C.recov = parseProp(at(R, "sensor.sipm.recovery_time"), "sensor.sipm.recovery_time", Kind::TIME);
  C.pmtGain = parseProp(at(R, "sensor.pmt.gain"), "sensor.pmt.gain", Kind::NONE);
  // basic timing / electronics (optional; 0 when absent or unknown)
  C.transit = parseProp(at(R, "sensor.timing.transit_time"), "sensor.timing.transit_time", Kind::TIME);
  C.jitter  = parseProp(at(R, "sensor.timing.time_jitter"), "sensor.timing.time_jitter", Kind::TIME);
  C.noise   = parseProp(at(R, "sensor.electronics.noise_rms"), "sensor.electronics.noise_rms", Kind::ELEC);
  range(C.transit, 0.0, 1e5, "transit time [ns]"); range(C.jitter, 0.0, 1e3, "time jitter [ns]"); range(C.noise, 0.0, 1e9, "noise rms [electrons]");
  { std::string pe = strOf(at(R, "sensor.pmt.enabled"), "false");
    if (pe != "true" && pe != "false") E("sensor.pmt.enabled", "must be true or false");
    C.pmtEnabled = (pe == "true");
    if (C.sType == "PMT" && !C.pmtEnabled) E("sensor.pmt.enabled", "sensor.type is PMT but pmt.enabled is false");
    if (C.sType == "SiPM" && C.pmtEnabled) W("sensor.pmt.enabled", "true, but sensor.type is SiPM: PMT response is not used"); }
  if (C.sType == "SiPM") {
    if (C.sipmSat) { need("photodetection", C.cells, "sensor.sipm.microcell_count (model_saturation: true)"); range(C.cells, 1, 1e8, "microcell count"); }
    if (C.sipmSat && C.sipmRec) { need("photodetection", C.recov, "sensor.sipm.recovery_time (model_recovery: true)"); range(C.recov, 0.0, 1e5, "recovery time [ns]", false); }
    if (C.sipmRec && !C.sipmSat) W("sensor.sipm.model_recovery", "recovery requires model_saturation: true; it will be ignored");
    if (C.gain.known) range(C.gain, 1.0, 1e9, "gain", true);
  } else if (C.sType == "PMT") {
    if (C.pmtGain.known) range(C.pmtGain, 1.0, 1e10, "gain", true);
  }

  // ---- geometry cross-checks ------------------------------------------------------------------
  if (C.dx > 0 && C.sW > 0 && !C.sFace.empty()) {
    double hu = 0, hv = 0;   // crystal in-plane dimensions for this face
    char ax = C.sFace[1];
    if (ax == 'z') { hu = C.dx; hv = C.dy; } else if (ax == 'x') { hu = C.dy; hv = C.dz; } else { hu = C.dz; hv = C.dx; }
    if (std::fabs(C.sOffU) + C.sW / 2 > hu / 2 + 1e-9 || std::fabs(C.sOffV) + C.sH / 2 > hv / 2 + 1e-9)
      E("sensor.active_area", "sensor rectangle (with offsets) does not fit on face " + C.sFace + " (face is " + std::to_string(hu) + " x " + std::to_string(hv) + " mm)");
  }
  if (C.dx > 0 && C.srcType == "pencil_beam") {
    // is the source inside the crystal? allowed (volume source) - just inform.
    if (std::fabs(C.srcPos.x()) < C.dx / 2 && std::fabs(C.srcPos.y()) < C.dy / 2 && std::fabs(C.srcPos.z()) < C.dz / 2)
      I("source.position", "source lies INSIDE the crystal (volume source)");
  }

  // ---- stage summary ------------------------------------------------------------------------------
  g_stage["gamma_screening"].ok = g_stage["gamma_screening"].ok && !C.elems.empty() && C.dx > 0 && C.events > 0;
  bool anyErr = false; for (auto& i : g_issues) if (i.sev == 'E') anyErr = true;
  return !anyErr;
}

// Decide effective mode (needs a valid config). Returns false when 'full' is demanded but not possible.
static bool DecideMode() {
  bool full = g_stage["scintillation"].ok && g_stage["optical_transport"].ok && g_stage["photodetection"].ok;
  if (C.modeRequested == "screening") { C.modeEffective = "gamma_screening"; return true; }
  if (C.modeRequested == "full") { C.modeEffective = "full"; return full; }
  C.modeEffective = full ? "full" : "gamma_screening";
  return true;
}

// ============================================================================
// 3. MATERIALS, GEOMETRY, PHYSICS
// ============================================================================
static const double HC_EV_NM = 1239.84198;   // h*c in eV*nm

struct EV { std::vector<double> E, y; };     // photon energy [G4 internal units] ascending, values
static double g_wlLo = 200, g_wlHi = 1200;   // wavelength span [nm] over which constant optical properties are defined
static void extendSpan(const Prop& p) {
  if (p.hasTable) { g_wlLo = std::min(g_wlLo, p.tx.front() * 0.98); g_wlHi = std::max(g_wlHi, p.tx.back() * 1.02); }
}
static EV constVec(double v) {
  EV r; r.E = {HC_EV_NM / g_wlHi * eV, HC_EV_NM / g_wlLo * eV}; r.y = {v, v}; return r;
}
static EV tableVec(const std::vector<double>& wl, const std::vector<double>& y, double yScale = 1.0) {
  EV r;
  for (int i = (int)wl.size() - 1; i >= 0; --i) { r.E.push_back(HC_EV_NM / wl[i] * eV); r.y.push_back(y[i] * yScale); }
  return r;
}
static EV propVec(const Prop& p, double scaleUnits = 1.0) {
  if (p.hasTable) return tableVec(p.tx, p.ty, scaleUnits);
  return constVec(p.value * scaleUnits);
}
// intensity spectrum (wavelength basis -> per-energy basis via Jacobian lambda^2)
static EV spectrumVec(const std::vector<double>& wl, const std::vector<double>& I, const std::string& basis) {
  std::vector<double> y(I.size());
  for (size_t i = 0; i < I.size(); ++i) y[i] = (basis == "per_wavelength") ? I[i] * wl[i] * wl[i] : I[i];
  double mx = *std::max_element(y.begin(), y.end());
  if (mx <= 0) mx = 1;
  for (auto& v : y) v /= mx;
  return tableVec(wl, y);
}
static void emissionPoints(std::vector<double>& wl, std::vector<double>& I, std::string& basis) {
  basis = "per_wavelength";
  if (C.specMode == "table") { wl = C.specTable.tx; I = C.specTable.ty; basis = C.specBasis; return; }
  double pk = C.peakWL.value;
  if (C.specMode == "gaussian") {
    double sg = C.fwhm.value / 2.354820045;
    const int N = 161;
    for (int i = 0; i < N; ++i) { double x = pk - 5 * sg + 10 * sg * i / (N - 1); if (x <= 1) continue; wl.push_back(x); I.push_back(std::exp(-0.5 * std::pow((x - pk) / sg, 2))); }
  } else {   // monochromatic: narrow triangle (width +-0.05 nm) at the peak
    wl = {pk - 0.05, pk, pk + 0.05}; I = {0.0, 1.0, 0.0};
  }
}
static double interpTable(const std::vector<double>& x, const std::vector<double>& y, double xv) {
  if (xv <= x.front()) return y.front();
  if (xv >= x.back()) return y.back();
  size_t i = std::upper_bound(x.begin(), x.end(), xv) - x.begin();
  double f = (xv - x[i - 1]) / (x[i] - x[i - 1]); return y[i - 1] + f * (y[i] - y[i - 1]);
}
static double pdeAt(double wl_nm) { return C.pde.hasTable ? interpTable(C.pde.tx, C.pde.ty, wl_nm) : C.pde.value; }

// ---- material summary for JSON ----------------------------------------------
struct MatElem { std::string sym; int Z; double Amol, atoms, massFrac; };
static std::vector<MatElem> g_matElems;
static double g_matRadLen_mm = 0, g_matElecDens_cm3 = 0, g_matDensity = 0, g_matTemp = 0;

static G4Material* BuildCrystalMaterial() {
  auto* nm = G4NistManager::Instance();
  double tot = 0; std::vector<double> m;
  std::vector<G4Element*> els;
  for (auto& e : C.elems) {
    G4Element* el = nm->FindOrBuildElement(e.first);
    els.push_back(el);
    double A = el->GetA() / (g / mole);
    m.push_back(e.second * A); tot += m.back();
    g_matElems.push_back({e.first, (int)el->GetZ(), A, e.second, 0});
  }
  G4State st = C.state == "solid" ? kStateSolid : (C.state == "liquid" ? kStateLiquid : kStateGas);
  auto* mat = new G4Material(C.matName, C.density.value * g / cm3, (G4int)els.size(), st, C.temperature.value * kelvin);
  for (size_t i = 0; i < els.size(); ++i) { mat->AddElement(els[i], m[i] / tot); g_matElems[i].massFrac = m[i] / tot; }
  g_matRadLen_mm = mat->GetRadlen() / mm;
  g_matElecDens_cm3 = mat->GetTotNbOfElectPerVolume() * cm3;
  g_matDensity = mat->GetDensity() / (g / cm3);
  g_matTemp = mat->GetTemperature() / kelvin;
  return mat;
}
static G4Material* Derive(const std::string& name, const std::string& nist) {
  const G4Material* b = G4NistManager::Instance()->FindOrBuildMaterial(nist);
  return new G4Material(name, b->GetDensity(), b, b->GetState(), b->GetTemperature(), b->GetPressure());
}

static bool g_full = false;
struct Geo {
  G4VPhysicalVolume *world = nullptr, *crystal = nullptr, *coupling = nullptr, *sensor = nullptr;
  double hx = 0, hy = 0, hz = 0, worldHalf = 0;
  int axN = 2, axU = 0, axV = 1; double sgn = 1;
  G4ThreeVector sensorCenter, couplingCenter;
} GEO;

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    GEO.hx = C.dx / 2; GEO.hy = C.dy / 2; GEO.hz = C.dz / 2;
    G4Material* crystalMat = BuildCrystalMaterial();
    G4Material* worldMat = Derive("SurroundingMedium", C.surrMat);
    double tC = 0, th = 0;
    char ax = C.sFace.size() > 1 ? C.sFace[1] : 'z';
    GEO.sgn = (C.sFace.size() > 0 && C.sFace[0] == '-') ? -1 : 1;
    if (ax == 'z') { GEO.axN = 2; GEO.axU = 0; GEO.axV = 1; } else if (ax == 'x') { GEO.axN = 0; GEO.axU = 1; GEO.axV = 2; } else { GEO.axN = 1; GEO.axU = 2; GEO.axV = 0; }
    if (g_full) { tC = (C.coupType == "layer") ? C.coupT.value : 0.0; th = C.sThick; }
    double ext = std::max({GEO.hx, GEO.hy, GEO.hz}) + tC + th;
    double smax = std::max({std::fabs(C.srcPos.x()), std::fabs(C.srcPos.y()), std::fabs(C.srcPos.z())});
    GEO.worldHalf = 1.5 * std::max(ext, smax) + 10.0;
    auto* wS = new G4Box("World", GEO.worldHalf * mm, GEO.worldHalf * mm, GEO.worldHalf * mm);
    auto* wL = new G4LogicalVolume(wS, worldMat, "World");
    GEO.world = new G4PVPlacement(nullptr, {}, wL, "World", nullptr, false, 0, false);
    auto* cS = new G4Box("Crystal", GEO.hx * mm, GEO.hy * mm, GEO.hz * mm);
    auto* cL = new G4LogicalVolume(cS, crystalMat, "Crystal");
    GEO.crystal = new G4PVPlacement(nullptr, {}, cL, "Crystal", wL, false, 0, false);
    auto* va = new G4VisAttributes(G4Colour(0.2, 0.8, 0.9, 0.4)); va->SetForceSolid(true); cL->SetVisAttributes(va);
    wL->SetVisAttributes(G4VisAttributes::GetInvisible());

    if (!g_full) return GEO.world;

    // ---- coupling + sensor on the chosen face --------------------------------------------
    double half[3] = {GEO.hx, GEO.hy, GEO.hz};
    auto vec = [&](double n, double u, double v) { G4ThreeVector r; r[GEO.axN] = n; r[GEO.axU] = u; r[GEO.axV] = v; return r; };
    if (C.coupType == "layer") {
      G4ThreeVector h = vec(tC / 2, C.sW / 2, C.sH / 2);
      auto* cpS = new G4Box("Coupling", h.x() * mm, h.y() * mm, h.z() * mm);
      auto* cpL = new G4LogicalVolume(cpS, Derive("CouplingMedium", C.coupMat), "Coupling");
      GEO.couplingCenter = vec(GEO.sgn * (half[GEO.axN] + tC / 2), C.sOffU, C.sOffV);
      GEO.coupling = new G4PVPlacement(nullptr, GEO.couplingCenter * mm, cpL, "Coupling", wL, false, 0, false);
      auto* v2 = new G4VisAttributes(G4Colour(0.9, 0.9, 0.2, 0.6)); v2->SetForceSolid(true); cpL->SetVisAttributes(v2);
      // optical properties of the coupling medium
      auto* mt = new G4MaterialPropertiesTable();
      EV n = propVec(C.coupN); mt->AddProperty("RINDEX", n.E, n.y);
      cpL->GetMaterial()->SetMaterialPropertiesTable(mt);
    }
    {
      G4ThreeVector h = vec(th / 2, C.sW / 2, C.sH / 2);
      auto* sS = new G4Box("Sensor", h.x() * mm, h.y() * mm, h.z() * mm);
      auto* sL = new G4LogicalVolume(sS, Derive("SensorWindow", C.winMat), "Sensor");
      GEO.sensorCenter = vec(GEO.sgn * (half[GEO.axN] + tC + th / 2), C.sOffU, C.sOffV);
      GEO.sensor = new G4PVPlacement(nullptr, GEO.sensorCenter * mm, sL, "Sensor", wL, false, 0, false);
      auto* v3 = new G4VisAttributes(G4Colour(0.9, 0.1, 0.1, 0.8)); v3->SetForceSolid(true); sL->SetVisAttributes(v3);
      auto* mt = new G4MaterialPropertiesTable();
      EV n = propVec(C.winN); mt->AddProperty("RINDEX", n.E, n.y);
      sL->GetMaterial()->SetMaterialPropertiesTable(mt);
    }
    { auto* mt = new G4MaterialPropertiesTable();
      EV n = propVec(C.surrN); mt->AddProperty("RINDEX", n.E, n.y);
      worldMat->SetMaterialPropertiesTable(mt); }

    // ---- crystal optical + scintillation properties -----------------------------------------------
    auto* mt = new G4MaterialPropertiesTable();
    { EV n = propVec(C.nCrystal); mt->AddProperty("RINDEX", n.E, n.y); }
    { EV a = C.absLen.infinite ? constVec(1e30) : propVec(C.absLen, mm);
      if (C.absLen.hasTable) { for (auto& v : a.y) v = std::min(v, 1e30 * mm); }
      mt->AddProperty("ABSLENGTH", a.E, a.y); }
    if (C.rayEnabled) { EV r = C.rayLen.infinite ? constVec(1e30) : propVec(C.rayLen, mm); mt->AddProperty("RAYLEIGH", r.E, r.y); }
    if (C.mieEnabled) {
      EV r = C.mieLen.infinite ? constVec(1e30) : propVec(C.mieLen, mm); mt->AddProperty("MIEHG", r.E, r.y);
      mt->AddConstProperty("MIEHG_FORWARD", C.mieFwd.value);
      mt->AddConstProperty("MIEHG_BACKWARD", C.mieBwd.value);
      mt->AddConstProperty("MIEHG_FORWARD_RATIO", C.mieRatio.value);
    }
    {
      std::vector<double> wl, I; std::string basis; emissionPoints(wl, I, basis);
      EV sp = spectrumVec(wl, I, basis);
      mt->AddConstProperty("SCINTILLATIONYIELD", C.lightYield.value / MeV);
      mt->AddConstProperty("RESOLUTIONSCALE", C.resScale.value);
      for (size_t i = 0; i < C.comps.size(); ++i) {
        std::string k = std::to_string(i + 1);
        mt->AddProperty("SCINTILLATIONCOMPONENT" + k, sp.E, sp.y);
        mt->AddConstProperty("SCINTILLATIONTIMECONSTANT" + k, C.comps[i].tau.value * ns);
        mt->AddConstProperty("SCINTILLATIONYIELD" + k, C.comps[i].frac.value);
        mt->AddConstProperty("SCINTILLATIONRISETIME" + k, (C.riseTime.known ? C.riseTime.value : 0.0) * ns);
      }
    }
    if (C.wlsEnabled) {
      EV a = propVec(C.wlsAbs, mm); mt->AddProperty("WLSABSLENGTH", a.E, a.y);
      EV e = spectrumVec(C.wlsEm.tx, C.wlsEm.ty, C.wlsBasis); mt->AddProperty("WLSCOMPONENT", e.E, e.y);
      mt->AddConstProperty("WLSMEANNUMBERPHOTONS", C.wlsMean.value);
      mt->AddConstProperty("WLSTIMECONSTANT", C.wlsTau.value * ns);
    }
    crystalMat->SetMaterialPropertiesTable(mt);

    // ---- surface between crystal and surrounding medium ---------------------------------------------
    if (C.reflType != "none" || C.finish == "ground") {
      auto* os = new G4OpticalSurface("CrystalSurface");
      os->SetType(dielectric_dielectric);
      os->SetModel(unified);
      auto* smt = new G4MaterialPropertiesTable();
      if (C.reflType == "specular") os->SetFinish(polishedfrontpainted);
      else if (C.reflType == "diffuse") os->SetFinish(groundfrontpainted);
      else os->SetFinish(ground);
      if (C.reflType != "none") { EV r = propVec(C.reflectivity); smt->AddProperty("REFLECTIVITY", r.E, r.y); }
      if (C.reflType == "none" && C.finish == "ground") {
        os->SetSigmaAlpha(C.sigmaAlpha.value * deg);
        auto lobe = [&](const char* key, const Prop& p, const char* label) {
          double v = 0; if (p.known) v = p.value; else g_assumptionsApplied.push_back(std::string("surface.unified_constants.") + label + " not provided: set to 0 (remaining reflection probability is Lambertian in the Geant4 unified model)");
          EV c = constVec(v); smt->AddProperty(key, c.E, c.y); };
        lobe("SPECULARLOBECONSTANT", C.lobeSpec, "specular_lobe");
        lobe("SPECULARSPIKECONSTANT", C.lobeSpike, "specular_spike");
        lobe("BACKSCATTERCONSTANT", C.lobeBack, "backscatter");
      }
      os->SetMaterialPropertiesTable(smt);
      new G4LogicalBorderSurface("CrystalSurface", GEO.crystal, GEO.world, os);
    }
    return GEO.world;
  }
};

static G4VModularPhysicsList* BuildPhysics() {
  G4PhysListFactory f; f.SetVerbose(0);
  if (!f.IsReferencePhysList(C.physList)) { std::cerr << "ERROR: unknown physics list '" << C.physList << "'\n"; std::exit(2); }
  G4VModularPhysicsList* pl = f.GetReferencePhysList(C.physList);
  pl->SetDefaultCutValue(C.cutMM * mm);
  pl->SetVerboseLevel(0);
  if (g_full) {
    auto* p = G4OpticalParameters::Instance();
    p->SetProcessActivation("Cerenkov", false);            // Cerenkov light is NOT modelled
    p->SetProcessActivation("Scintillation", true);
    p->SetProcessActivation("OpAbsorption", true);
    p->SetProcessActivation("OpRayleigh", C.rayEnabled);
    p->SetProcessActivation("OpMieHG", C.mieEnabled);
    p->SetProcessActivation("OpBoundary", true);
    p->SetProcessActivation("OpWLS", C.wlsEnabled);
    p->SetProcessActivation("OpWLS2", false);
    p->SetScintByParticleType(false);                       // same yield for every ionising particle (no nonproportionality)
    p->SetScintTrackSecondariesFirst(true);
    p->SetScintFiniteRiseTime(true);
    p->SetBoundaryInvokeSD(false);
    pl->RegisterPhysics(new G4OpticalPhysics());
  }
  return pl;
}

// ============================================================================
// 4. RUN DATA, HISTOGRAMS, OUTPUT WRITERS
// ============================================================================
struct Hist1D {
  double lo = 0, hi = 1; int n = 1; std::vector<double> c; double under = 0, over = 0, cnt = 0, sum = 0;
  void init(double l, double h, int nb) { lo = l; hi = h; n = nb; c.assign(nb, 0.0); under = over = cnt = sum = 0; }
  void fill(double x) {
    cnt += 1; sum += x;
    if (x < lo) { under += 1; return; }
    if (x >= hi) { over += 1; return; }
    c[(int)((x - lo) / (hi - lo) * n)] += 1;
  }
};
struct Hist2D {
  double xlo = 0, xhi = 1, ylo = 0, yhi = 1; int nx = 1, ny = 1; std::vector<double> c;
  void init(double a, double b, double d, double e, int NX, int NY) { xlo = a; xhi = b; ylo = d; yhi = e; nx = NX; ny = NY; c.assign((size_t)NX * NY, 0.0); }
  void fill(double x, double y, double w) {
    if (x < xlo || x >= xhi || y < ylo || y >= yhi) return;
    int i = (int)((x - xlo) / (xhi - xlo) * nx), j = (int)((y - ylo) / (yhi - ylo) * ny);
    c[(size_t)i * ny + j] += w;
  }
};
enum Term { T_DET = 0, T_NODET, T_BULK, T_SURF, T_ESC, T_WLS, T_UNCL, T_N };
static const char* kTermName[T_N] = {"sensor_detected", "sensor_not_detected", "bulk_absorbed", "surface_absorbed", "escaped", "wls_converted", "unclassified"};
static const char* kFaceName[7] = {"+x", "-x", "+y", "-y", "+z", "-z", "coupling_edge"};

struct Evt {
  int id = 0;
  double Eprim = 0, Edep = 0, Eout = 0, Ereentry = 0, t0 = -1;
  bool primEntered = false, interacted = false; std::string firstProc = "none";
  G4ThreeVector firstPos, edepSum;   // edepSum = sum(edep * position)
  long nScint = 0, nWLS = 0, nBoundary = 0, nRay = 0, nMie = 0, nRefl = 0, nIncident = 0;
  long term[T_N] = {0};
  std::vector<double> detTimes;
  long cellsFired = 0, cellsLost = 0; double charge = 0, tFirst = -1, tMean = -1;
  void reset(int i) { *this = Evt(); id = i; }
};
static Evt EVT;

struct PhotonRec {
  long track = 0, parent = 0; int creator = 0; double wl = 0, tEmit = 0, tEnd = 0, path = 0;
  G4ThreeVector pEmit, pEnd; int term = -1, face = -1; long nBound = 0, nRefl = 0, nRay = 0, nMie = 0;
};
static PhotonRec PH;

struct RunData {
  std::ofstream evOut, phOut;
  long nEvents = 0, nInteracted = 0, nEdep = 0, nPrimEntered = 0;
  double sumEdep = 0, maxAbsClosure = 0;
  std::map<std::string, long> firstProc;
  long nScint = 0, nWLS = 0, nBoundary = 0, nRay = 0, nMie = 0, nRefl = 0, nIncident = 0, nDet = 0;
  long term[T_N] = {0};
  long faceLoss[2][7] = {{0}};   // [0]=surface_absorbed, [1]=escaped; index 6 = lateral edge of the coupling layer
  long long cellsFired = 0, cellsLost = 0; double charge = 0;
  Hist1D hWl, hTemit, hTarr, hTdet; Hist2D mapXY, mapXZ, mapYZ;
  long phRows = 0; bool phTruncated = false;
  double attLen_mm = -1, muPhot = -1, muCompt = -1, muConv = -1, muRayl = -1, chord_mm = -1, expectedPint = -1;
  double wall_s = 0; std::string timestamp;
} RD;

static std::string jesc(const std::string& s) {
  std::string o;
  for (char ch : s) {
    switch (ch) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
      default: if ((unsigned char)ch < 0x20) { char b[8]; std::snprintf(b, 8, "\\u%04x", ch); o += b; } else o += ch; }
  }
  return o;
}
static std::string jn(double v) {
  if (std::isnan(v)) return "null";
  if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
  std::ostringstream o; o << std::setprecision(10) << v; return o.str();
}
static std::string js(const std::string& s) { return "\"" + jesc(s) + "\""; }
static std::string jvec(const std::vector<double>& v) { std::string o = "["; for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + jn(v[i]); return o + "]"; }
static std::string jhist(const Hist1D& h) {
  return "{\"lo\":" + jn(h.lo) + ",\"hi\":" + jn(h.hi) + ",\"bins\":" + std::to_string(h.n) + ",\"counts\":" + jvec(h.c) + ",\"underflow\":" + jn(h.under) + ",\"overflow\":" + jn(h.over) + ",\"entries\":" + jn(h.cnt) + "}";
}
static std::string jhist2(const Hist2D& h) {
  std::string o = "{\"xlo\":" + jn(h.xlo) + ",\"xhi\":" + jn(h.xhi) + ",\"ylo\":" + jn(h.ylo) + ",\"yhi\":" + jn(h.yhi) + ",\"nx\":" + std::to_string(h.nx) + ",\"ny\":" + std::to_string(h.ny) + ",\"counts\":[";
  for (int i = 0; i < h.nx; ++i) { std::vector<double> row(h.c.begin() + (size_t)i * h.ny, h.c.begin() + (size_t)(i + 1) * h.ny); o += (i ? "," : "") + jvec(row); }
  return o + "]}";
}
// YAML -> JSON (faithful echo of the configuration for reproducibility)
static std::string yamlToJson(const YAML::Node& n) {
  if (!n.IsDefined() || n.IsNull()) return "null";
  if (n.IsScalar()) {
    std::string s = n.Scalar(); double v;
    if (n.Tag() != "!" ) {
      if (s == "true" || s == "false") return s;
      if (s == "null" || s == "~") return "null";
      if (parseDouble(s, v)) return jn(v);
    }
    return js(s);
  }
  if (n.IsSequence()) { std::string o = "["; bool f = true; for (const auto& c : n) { o += (f ? "" : ",") + yamlToJson(c); f = false; } return o + "]"; }
  std::string o = "{"; bool f = true;
  for (auto it = n.begin(); it != n.end(); ++it) { o += (f ? "" : ",") + js(it->first.as<std::string>()) + ":" + yamlToJson(it->second); f = false; }
  return o + "}";
}
static std::string nowIso() {
  std::time_t t = std::time(nullptr); char b[40]; std::strftime(b, 40, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t)); return b;
}

// Explicit microcell SiPM model: each detected photon fires a uniformly random cell;
// a cell that fired is dead for recovery_time (full dead-time model), photons hitting a dead cell are lost.
static void SipmEval(std::vector<double> t, long& fired, long& lost) {
  fired = 0; lost = 0;
  if (!C.sipmSat) { fired = (long)t.size(); return; }
  std::sort(t.begin(), t.end());
  long N = (long)C.cells.value;
  std::unordered_map<long, double> last;
  for (double ti : t) {
    long cell = std::min(N - 1, (long)(G4UniformRand() * N));
    auto it = last.find(cell);
    if (it == last.end()) { last[cell] = ti; ++fired; }
    else if (C.sipmRec && (ti - it->second) >= C.recov.value) { it->second = ti; ++fired; }
    else ++lost;
  }
}

// ============================================================================
// 5. USER ACTIONS
// ============================================================================
class PrimaryGen : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGen() : gun(1) { gun.SetParticleDefinition(G4Gamma::Gamma()); gun.SetParticleEnergy(C.eGammaKeV * keV); }
  void GeneratePrimaries(G4Event* ev) override {
    G4ThreeVector d = C.srcDir;
    if (C.srcType != "pencil_beam") {
      double cmin = std::cos(C.coneDeg * deg);
      double ct = 1.0 - G4UniformRand() * (1.0 - cmin), st = std::sqrt(std::max(0.0, 1 - ct * ct)), ph = twopi * G4UniformRand();
      d = G4ThreeVector(st * std::cos(ph), st * std::sin(ph), ct); d.rotateUz(C.srcDir);
    }
    gun.SetParticlePosition(C.srcPos * mm); gun.SetParticleMomentumDirection(d); gun.GeneratePrimaryVertex(ev);
  }
 private:
  G4ParticleGun gun;
};

static int faceOf(const G4ThreeVector& p) {
  double d[6] = {GEO.hx - p.x(), GEO.hx + p.x(), GEO.hy - p.y(), GEO.hy + p.y(), GEO.hz - p.z(), GEO.hz + p.z()};
  return (int)(std::min_element(d, d + 6) - d);
}

class TrackAct : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* t) override {
    if (t->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition()) return;
    PH = PhotonRec();
    PH.track = t->GetTrackID(); PH.parent = t->GetParentID();
    const G4VProcess* cp = t->GetCreatorProcess();
    PH.creator = (cp && cp->GetProcessName() == "OpWLS") ? 1 : 0;
    PH.wl = HC_EV_NM / (t->GetKineticEnergy() / eV);
    double t0 = EVT.t0 < 0 ? 0 : EVT.t0;
    PH.tEmit = t->GetGlobalTime() / ns - t0;
    PH.pEmit = t->GetPosition() / mm;
    if (PH.creator == 0) { EVT.nScint++; RD.hWl.fill(PH.wl); RD.hTemit.fill(PH.tEmit); } else EVT.nWLS++;
  }
  void PostUserTrackingAction(const G4Track* t) override {
    if (t->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition()) return;
    if (PH.term < 0) { PH.term = T_UNCL; PH.pEnd = t->GetPosition() / mm; PH.tEnd = t->GetGlobalTime() / ns - (EVT.t0 < 0 ? 0 : EVT.t0); }
    EVT.term[PH.term]++;
    EVT.nBoundary += PH.nBound > 0; EVT.nRay += PH.nRay; EVT.nMie += PH.nMie; EVT.nRefl += PH.nRefl;
    if (PH.term == T_SURF || PH.term == T_ESC) if (PH.face >= 0) RD.faceLoss[PH.term == T_ESC][PH.face]++;
    bool atSensor = (PH.term == T_DET || PH.term == T_NODET);
    bool log = (C.photonLog == "all") || (C.photonLog == "sensor" && atSensor);
    if (log && RD.phOut.is_open()) {
      if (RD.phRows >= C.maxPhotonLog) RD.phTruncated = true;
      else {
        RD.phOut << EVT.id << ',' << PH.track << ',' << PH.parent << ',' << (PH.creator ? "OpWLS" : "Scintillation") << ',' << PH.wl << ',' << PH.tEmit << ','
                 << PH.pEmit.x() << ',' << PH.pEmit.y() << ',' << PH.pEmit.z() << ',' << kTermName[PH.term] << ',' << PH.tEnd << ','
                 << PH.pEnd.x() << ',' << PH.pEnd.y() << ',' << PH.pEnd.z() << ',' << (PH.face >= 0 ? kFaceName[PH.face] : "none") << ',' << PH.nBound << ',' << PH.nRefl << ','
                 << PH.nRay << ',' << PH.nMie << ',' << PH.path << ',' << (PH.term == T_DET ? 1 : 0) << '\n';
        RD.phRows++;
      }
    }
  }
};

class StepAct : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4Track* trk = step->GetTrack();
    if (trk->GetDefinition() == G4OpticalPhoton::OpticalPhotonDefinition()) { OpticalStep(step, trk); return; }
    const G4StepPoint* pre = step->GetPreStepPoint(); const G4StepPoint* post = step->GetPostStepPoint();
    const G4VPhysicalVolume* prePV = pre->GetPhysicalVolume(); const G4VPhysicalVolume* postPV = post->GetPhysicalVolume();
    bool preIn = prePV == GEO.crystal, postIn = postPV == GEO.crystal;
    bool primary = trk->GetTrackID() == 1;
    // energy flow across the crystal surface
    if (primary && trk->GetCurrentStepNumber() == 1 && preIn) { EVT.Eprim = pre->GetKineticEnergy() / keV; EVT.primEntered = true; }
    if (preIn && !postIn && post->GetStepStatus() == fGeomBoundary) EVT.Eout += post->GetKineticEnergy() / keV;
    if (!preIn && postIn) {
      double ke = post->GetKineticEnergy() / keV;
      if (primary && !EVT.primEntered) { EVT.Eprim = ke; EVT.primEntered = true; } else EVT.Ereentry += ke;
    }
    if (!preIn) return;
    // energy deposition
    double ed = step->GetTotalEnergyDeposit() / keV;
    if (ed > 0) {
      if (EVT.t0 < 0) EVT.t0 = pre->GetGlobalTime() / ns;
      EVT.Edep += ed;
      G4ThreeVector mid = 0.5 * (pre->GetPosition() + post->GetPosition()) / mm;
      EVT.edepSum += ed * mid;
      RD.mapXY.fill(mid.x(), mid.y(), ed); RD.mapXZ.fill(mid.x(), mid.z(), ed); RD.mapYZ.fill(mid.y(), mid.z(), ed);
    }
    // first interaction of the primary gamma inside the crystal
    if (primary && !EVT.interacted && post->GetProcessDefinedStep()) {
      const std::string& pn = post->GetProcessDefinedStep()->GetProcessName();
      if (pn != "Transportation" && pn != "CoupledTransportation" && pn != "Scintillation" && pn.rfind("Op", 0) != 0 && post->GetStepStatus() != fWorldBoundary) {
        EVT.interacted = true; EVT.firstProc = pn; EVT.firstPos = post->GetPosition() / mm;
      }
    }
  }

 private:
  G4OpBoundaryProcess* bp = nullptr;
  void kill(const G4Track* t) { const_cast<G4Track*>(t)->SetTrackStatus(fStopAndKill); }
  void finish(const G4Step* s, int term, int face) {
    PH.term = term; PH.face = face; PH.pEnd = s->GetPostStepPoint()->GetPosition() / mm;
    PH.tEnd = s->GetPostStepPoint()->GetGlobalTime() / ns - (EVT.t0 < 0 ? 0 : EVT.t0);
  }
  void OpticalStep(const G4Step* step, const G4Track* trk) {
    if (!bp) {
      auto* pm = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
      auto* pl = pm->GetProcessList();
      for (size_t i = 0; i < pl->size(); ++i) if ((*pl)[i]->GetProcessName() == "OpBoundary") bp = dynamic_cast<G4OpBoundaryProcess*>((*pl)[i]);
    }
    const G4StepPoint* pre = step->GetPreStepPoint(); const G4StepPoint* post = step->GetPostStepPoint();
    PH.path += step->GetStepLength() / mm;
    const G4VProcess* proc = post->GetProcessDefinedStep();
    std::string pn = proc ? proc->GetProcessName() : "";
    if (pn == "OpRayleigh") PH.nRay++;
    else if (pn == "OpMieHG") PH.nMie++;
    else if (pn == "OpAbsorption") { finish(step, T_BULK, -1); return; }
    else if (pn == "OpWLS") { finish(step, T_WLS, -1); return; }
    if (post->GetStepStatus() != fGeomBoundary || !bp) return;
    G4OpBoundaryProcessStatus st = bp->GetStatus();
    const G4VPhysicalVolume* postPV = post->GetPhysicalVolume();
    const G4VPhysicalVolume* prePV = pre->GetPhysicalVolume();
    switch (st) {
      case Absorption:
        PH.nBound++; finish(step, T_SURF, prePV == GEO.crystal ? faceOf(post->GetPosition() / mm) : -1); return;
      case FresnelReflection: case TotalInternalReflection: case LambertianReflection: case LobeReflection: case SpikeReflection: case BackScattering:
        PH.nBound++; PH.nRefl++; return;
      case FresnelRefraction: case Transmission:
        PH.nBound++;
        if (postPV == GEO.sensor && GEO.sensor) {
          EVT.nIncident++;
          double t = post->GetGlobalTime() / ns - (EVT.t0 < 0 ? 0 : EVT.t0);
          RD.hTarr.fill(t);
          bool det = G4UniformRand() < pdeAt(PH.wl);
          finish(step, det ? T_DET : T_NODET, -1);
          if (det) {
            double td = t + C.transit.value;
            if (C.jitter.value > 0) td += CLHEP::RandGauss::shoot(0.0, C.jitter.value);
            EVT.detTimes.push_back(td); RD.hTdet.fill(td);
          }
          kill(trk); return;
        }
        if (postPV == GEO.world) { finish(step, T_ESC, prePV == GEO.crystal ? faceOf(post->GetPosition() / mm) : (prePV == GEO.coupling && GEO.coupling ? 6 : -1)); kill(trk); return; }
        return;
      case NoRINDEX: PH.nBound++; return;   // photon is killed by Geant4 -> ends up 'unclassified' (indicates a configuration problem)
      default: return;
    }
  }
};

class EventAct : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event* ev) override { EVT.reset(ev->GetEventID()); }
  void EndOfEventAction(const G4Event*) override {
    Evt& e = EVT;
    double eg = C.eGammaKeV;
    double escNet = e.Eout - e.Ereentry;
    double closure = e.Eprim - e.Edep - escNet;
    G4ThreeVector cen = e.Edep > 0 ? e.edepSum / e.Edep : G4ThreeVector(0, 0, 0);
    // photodetection response
    if (g_full) {
      if (C.sType == "SiPM") { SipmEval(e.detTimes, e.cellsFired, e.cellsLost); e.charge = e.cellsFired * C.gain.value; }
      else { e.cellsFired = (long)e.detTimes.size(); e.charge = C.pmtGain.known ? e.detTimes.size() * C.pmtGain.value : std::nan(""); }
      if (C.noise.value > 0 && !std::isnan(e.charge)) e.charge += CLHEP::RandGauss::shoot(0.0, C.noise.value);
      if (!e.detTimes.empty()) {
        e.tFirst = *std::min_element(e.detTimes.begin(), e.detTimes.end());
        double s = 0; for (double t : e.detTimes) s += t; e.tMean = s / e.detTimes.size();
      }
    }
    auto& o = RD.evOut;
    o << e.id << ',' << eg << ',' << e.Eprim << ',' << e.Edep << ',' << escNet << ',' << closure << ',' << (e.interacted ? 1 : 0) << ',' << e.firstProc << ','
      << (e.Edep > 0 ? cen.x() : std::nan("")) << ',' << (e.Edep > 0 ? cen.y() : std::nan("")) << ',' << (e.Edep > 0 ? cen.z() : std::nan("")) << ','
      << (e.interacted ? e.firstPos.x() : std::nan("")) << ',' << (e.interacted ? e.firstPos.y() : std::nan("")) << ',' << (e.interacted ? e.firstPos.z() : std::nan(""));
    if (g_full) {
      o << ',' << e.nScint << ',' << e.nWLS << ',' << e.nBoundary << ',' << e.term[T_BULK] << ',' << e.term[T_SURF] << ',' << e.term[T_ESC] << ',' << e.term[T_WLS] << ',' << e.term[T_UNCL] << ','
        << e.nIncident << ',' << e.term[T_DET] << ',' << e.term[T_NODET] << ',' << e.nRay << ',' << e.nMie << ',' << e.nRefl << ',' << e.cellsFired << ',' << e.cellsLost << ',' << e.charge << ','
        << (!e.detTimes.empty() ? e.tFirst : std::nan("")) << ',' << (!e.detTimes.empty() ? e.tMean : std::nan(""));
    }
    o << '\n';
    // run totals
    RD.nEvents++; RD.nInteracted += e.interacted; RD.nEdep += e.Edep > 0; RD.nPrimEntered += e.primEntered; RD.sumEdep += e.Edep;
    RD.firstProc[e.firstProc]++;
    if (e.primEntered) RD.maxAbsClosure = std::max(RD.maxAbsClosure, std::fabs(closure));
    RD.nScint += e.nScint; RD.nWLS += e.nWLS; RD.nBoundary += e.nBoundary; RD.nRay += e.nRay; RD.nMie += e.nMie; RD.nRefl += e.nRefl;
    RD.nIncident += e.nIncident; RD.nDet += e.term[T_DET]; for (int i = 0; i < T_N; ++i) RD.term[i] += e.term[i];
    RD.cellsFired += e.cellsFired; RD.cellsLost += e.cellsLost; if (!std::isnan(e.charge)) RD.charge += e.charge;
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new PrimaryGen); SetUserAction(new EventAct); SetUserAction(new TrackAct); SetUserAction(new StepAct);
  }
};

// ---- gamma-ray analytic/Geant4-table calculations (valid calculations, not simulation) ------
static double ChordLength() {   // path length of the beam through the box (forward half-line); -1 if not a pencil beam
  if (C.srcType != "pencil_beam") return -1;
  double h[3] = {C.dx / 2, C.dy / 2, C.dz / 2}, p[3] = {C.srcPos.x(), C.srcPos.y(), C.srcPos.z()}, d[3] = {C.srcDir.x(), C.srcDir.y(), C.srcDir.z()};
  double tin = -1e30, tout = 1e30;
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(d[i]) < 1e-12) { if (std::fabs(p[i]) > h[i]) return 0; continue; }
    double a = (-h[i] - p[i]) / d[i], b = (h[i] - p[i]) / d[i];
    tin = std::max(tin, std::min(a, b)); tout = std::min(tout, std::max(a, b));
  }
  double t0 = std::max(tin, 0.0);
  return tout > t0 ? tout - t0 : 0.0;
}
static void ComputeGammaTables() {
  G4Material* mat = G4Material::GetMaterial(C.matName, false);
  if (!mat) return;
  G4EmCalculator calc;
  double E = C.eGammaKeV * keV;
  RD.attLen_mm = calc.ComputeGammaAttenuationLength(E, mat) / mm;
  auto mu = [&](const char* p) { return calc.ComputeCrossSectionPerVolume(E, G4Gamma::Gamma(), p, mat) * mm; };
  RD.muPhot = mu("phot"); RD.muCompt = mu("compt"); RD.muConv = mu("conv"); RD.muRayl = mu("Rayl");
  RD.chord_mm = ChordLength();
  if (RD.chord_mm >= 0 && RD.attLen_mm > 0) RD.expectedPint = 1.0 - std::exp(-RD.chord_mm / RD.attLen_mm);
}

static std::string stageJson() {
  std::ostringstream o; bool first = true; o << "{";
  for (const char* nm : {"gamma_screening", "scintillation", "optical_transport", "photodetection"}) {
    const StageInfo& s = g_stage[nm];
    bool ran = (std::string(nm) == "gamma_screening") || g_full;
    o << (first ? "" : ",") << js(nm) << ":{\"status\":" << js(ran ? "ran" : "not_run") << ",\"inputs_complete\":" << (s.ok ? "true" : "false") << ",\"missing_inputs\":[";
    for (size_t i = 0; i < s.missing.size(); ++i) o << (i ? "," : "") << js(s.missing[i]);
    o << "],\"reason\":" << js(ran ? "" : (C.modeRequested == "screening" ? "mode: screening requested by the user" : "required inputs missing (see missing_inputs)")) << "}";
    first = false;
  }
  return o.str() + "}";
}

static void WriteSummary(const std::string& path) {
  std::ostringstream o; o << std::setprecision(10);
  o << "{\n\"schema_version\":1,\n";
  o << "\"software\":{\"name\":" << js(SOFTWARE_NAME) << ",\"version\":" << js(SOFTWARE_VERSION) << ",\"geant4_version\":" << js(G4RunManager::GetRunManager()->GetVersionString()) << ",\"threading\":\"serial (single thread)\"},\n";
  o << "\"run\":{\"timestamp_utc\":" << js(RD.timestamp) << ",\"config_file\":" << js(g_configPath) << ",\"mode_requested\":" << js(C.modeRequested) << ",\"mode_effective\":" << js(C.modeEffective)
    << ",\"events\":" << RD.nEvents << ",\"seed\":" << C.seed << ",\"physics_list\":" << js(C.physList) << ",\"production_cut_mm\":" << jn(C.cutMM) << ",\"wall_time_s\":" << jn(RD.wall_s) << ",\"cli_overrides\":{";
  for (size_t i = 0; i < C.cliOverrides.size(); ++i) o << (i ? "," : "") << js(C.cliOverrides[i].first) << ":" << js(C.cliOverrides[i].second);
  o << "}},\n";
  o << "\"stages\":" << stageJson() << ",\n";
  o << "\"material\":{\"name\":" << js(C.matName) << ",\"physical_state\":" << js(C.state) << ",\"density_g_cm3\":" << jn(g_matDensity) << ",\"temperature_K\":" << jn(g_matTemp)
    << ",\"temperature_note\":\"recorded only; Geant4 gamma interaction physics does not depend on temperature\",\"radiation_length_mm\":" << jn(g_matRadLen_mm) << ",\"electron_density_per_cm3\":" << jn(g_matElecDens_cm3) << ",\"composition\":[";
  for (size_t i = 0; i < g_matElems.size(); ++i) { auto& e = g_matElems[i]; o << (i ? "," : "") << "{\"symbol\":" << js(e.sym) << ",\"Z\":" << e.Z << ",\"molar_mass_g_mol\":" << jn(e.Amol) << ",\"atoms_per_formula_unit\":" << jn(e.atoms) << ",\"mass_fraction\":" << jn(e.massFrac) << "}"; }
  o << "]},\n";
  o << "\"geometry\":{\"crystal_shape\":" << js(C.shape) << ",\"dimensions_mm\":[" << jn(C.dx) << "," << jn(C.dy) << "," << jn(C.dz) << "],\"volume_mm3\":" << jn(C.dx * C.dy * C.dz)
    << ",\"surrounding_material\":" << js(C.surrMat) << ",\"surface_finish\":" << js(C.finish) << ",\"reflector\":" << js(C.reflType);
  if (g_full) o << ",\"sensor_face\":" << js(C.sFace) << ",\"sensor_active_area_mm\":[" << jn(C.sW) << "," << jn(C.sH) << "],\"sensor_offset_mm\":[" << jn(C.sOffU) << "," << jn(C.sOffV) << "],\"coupling\":" << js(C.coupType);
  o << ",\"world_half_size_mm\":" << jn(GEO.worldHalf) << "},\n";
  o << "\"source\":{\"particle\":" << js(C.particle) << ",\"type\":" << js(C.srcType) << ",\"energy_keV\":" << jn(C.eGammaKeV) << ",\"position_mm\":[" << jn(C.srcPos.x()) << "," << jn(C.srcPos.y()) << "," << jn(C.srcPos.z())
    << "],\"direction\":[" << jn(C.srcDir.x()) << "," << jn(C.srcDir.y()) << "," << jn(C.srcDir.z()) << "],\"cone_half_angle_deg\":" << jn(C.coneDeg) << "},\n";
  if (g_full) {
    o << "\"scintillation_inputs\":{\"light_yield_photons_per_MeV\":" << jn(C.lightYield.value) << ",\"light_yield_status\":" << js(C.lightYield.status) << ",\"resolution_scale\":" << jn(C.resScale.value)
      << ",\"emission_peak_nm\":" << jn(C.peakWL.value) << ",\"emission_fwhm_nm\":" << (C.fwhm.known ? jn(C.fwhm.value) : "null") << ",\"spectrum_mode\":" << js(C.specMode)
      << ",\"rise_time_ns\":" << jn(C.riseTime.known ? C.riseTime.value : 0.0) << ",\"decay_components\":[";
    for (size_t i = 0; i < C.comps.size(); ++i) o << (i ? "," : "") << "{\"fraction\":" << jn(C.comps[i].frac.value) << ",\"decay_time_ns\":" << jn(C.comps[i].tau.value) << "}";
    std::vector<double> wl, I; std::string basis; emissionPoints(wl, I, basis);
    std::vector<double> Ipw(I.size());   // per-wavelength shape for plotting
    for (size_t i = 0; i < I.size(); ++i) Ipw[i] = basis == "per_wavelength" ? I[i] : I[i] / (wl[i] * wl[i]);
    double mx = *std::max_element(Ipw.begin(), Ipw.end()); for (auto& v : Ipw) v /= (mx > 0 ? mx : 1);
    o << "],\"emission_spectrum_used\":{\"wavelength_nm\":" << jvec(wl) << ",\"relative_intensity_per_wavelength\":" << jvec(Ipw) << "}},\n";
    auto tab = [&](const Prop& p) { std::string s = "{\"constant\":" + (p.hasTable ? std::string("null") : (p.infinite ? "\"inf\"" : jn(p.value))) + ",\"table_wavelength_nm\":" + jvec(p.tx) + ",\"table_values\":" + jvec(p.ty) + ",\"status\":" + js(p.status) + "}"; return s; };
    o << "\"optical_inputs\":{\"refractive_index\":" << tab(C.nCrystal) << ",\"absorption_length_mm\":" << tab(C.absLen)
      << ",\"surrounding_refractive_index\":" << tab(C.surrN) << ",\"rayleigh_enabled\":" << (C.rayEnabled ? "true" : "false") << ",\"rayleigh_length_mm\":" << (C.rayEnabled ? tab(C.rayLen) : "null")
      << ",\"mie_enabled\":" << (C.mieEnabled ? "true" : "false") << ",\"mie_length_mm\":" << (C.mieEnabled ? tab(C.mieLen) : "null") << ",\"wls_enabled\":" << (C.wlsEnabled ? "true" : "false")
      << ",\"reflector\":" << js(C.reflType) << ",\"reflectivity\":" << (C.reflType != "none" ? jn(C.reflectivity.value) : "null") << ",\"surface_finish_unwrapped\":" << js(C.finish)
      << ",\"sigma_alpha_deg\":" << (C.reflType == "none" && C.finish == "ground" ? jn(C.sigmaAlpha.value) : "null") << "},\n";
    o << "\"detector_inputs\":{\"type\":" << js(C.sType) << ",\"pde\":" << tab(C.pde) << ",\"window_refractive_index\":" << tab(C.winN) << ",\"coupling_refractive_index\":" << (C.coupType == "layer" ? tab(C.coupN) : "null")
      << ",\"coupling_thickness_mm\":" << (C.coupType == "layer" ? jn(C.coupT.value) : "null");
    if (C.sType == "SiPM") o << ",\"model_saturation\":" << (C.sipmSat ? "true" : "false") << ",\"model_recovery\":" << (C.sipmRec ? "true" : "false") << ",\"microcell_count\":" << (C.cells.known ? jn(C.cells.value) : "null")
      << ",\"gain\":" << (C.gain.known ? jn(C.gain.value) : "null") << ",\"recovery_time_ns\":" << (C.recov.known ? jn(C.recov.value) : "null");
    else o << ",\"gain\":" << (C.pmtGain.known ? jn(C.pmtGain.value) : "null");
    o << ",\"transit_time_ns\":" << jn(C.transit.value) << ",\"time_jitter_ns\":" << jn(C.jitter.value) << ",\"noise_rms_electrons\":" << jn(C.noise.value);
    o << "},\n";
  }
  o << "\"provenance\":[";
  for (size_t i = 0; i < g_prov.size(); ++i) { auto& p = g_prov[i]; o << (i ? "," : "") << "\n  {\"path\":" << js(p.path) << ",\"value\":" << js(p.valueText) << ",\"unit\":" << js(p.unit) << ",\"status\":" << js(p.status) << ",\"source\":" << js(p.source) << ",\"conditions\":" << js(p.conditions) << ",\"known\":" << (p.known ? "true" : "false") << "}"; }
  o << "\n],\n\"assumptions_applied\":[";
  for (size_t i = 0; i < g_assumptionsApplied.size(); ++i) o << (i ? "," : "") << js(g_assumptionsApplied[i]);
  o << "],\n";
  // models
  o << "\"physics_models\":{\"implemented\":[" << js("Gamma EM interactions via Geant4 reference physics list " + C.physList + " (photoelectric, Compton, Rayleigh, pair production where kinematically allowed)") << "," << js("Secondary e-/e+/gamma transport with production cut " + std::to_string(C.cutMM) + " mm");
  if (g_full) {
    o << "," << js(std::string("Scintillation: Geant4 G4Scintillation, yield linear in energy deposit; photon number Poisson (mean<=10) or Gaussian with sigma = resolution_scale*sqrt(mean) (mean>10)"))
      << "," << js("Emission time: finite rise time and " + std::to_string(C.comps.size()) + " exponential decay component(s)")
      << "," << js("Optical transport: bulk absorption, boundary process (Fresnel refraction/reflection, surface model " + C.surfModel + ")")
      << (C.rayEnabled ? "," + js("Rayleigh bulk scattering") : "") << (C.mieEnabled ? "," + js("Mie (Henyey-Greenstein) bulk scattering") : "") << (C.wlsEnabled ? "," + js("Bulk wavelength shifting") : "")
      << "," << js("Photodetection: wavelength-dependent PDE applied to each photon entering the sensor window (binomial photon counting)") << "," << js("Sensor timing: constant transit-time offset and Gaussian jitter per detected photon; Gaussian charge noise per event");
    if (C.sType == "SiPM" && C.sipmSat) o << "," << js(std::string("SiPM microcell saturation (random cell per detected photon)") + (C.sipmRec ? " with full dead-time recovery model" : ", no recovery"));
  }
  o << "],\"not_modelled\":[";
  std::vector<std::string> nm{"Birks quenching and scintillation nonproportionality (yield is linear in deposited energy)", "Cerenkov light", "Temperature dependence of any property", "Atomic fluorescence beyond what the production cut and physics list produce"};
  if (g_full) { nm.push_back("Electronics baseline, shaping, digitisation (only Gaussian charge noise is applied)"); nm.push_back("SiPM optical crosstalk, afterpulsing, dark counts, PMT transit-time spread and gain fluctuations");
    nm.push_back("Photons transmitted into the surrounding medium are discarded (counted as escaped); re-entry into the crystal is not modelled");
    nm.push_back("Absorption in the optical coupling layer and sensor window (treated as transparent)"); nm.push_back("Wavelength-dependent reflectivity of the wrapper (constant or tabulated value used as given)"); nm.push_back("Self-absorption/re-emission (reabsorption without re-emission is represented by the absorption length only)"); }
  else nm.push_back("ALL scintillation, optical transport and photodetection stages (not run in gamma_screening mode)");
  for (size_t i = 0; i < nm.size(); ++i) o << (i ? "," : "") << js(nm[i]);
  o << "]},\n";
  // totals
  o << "\"totals\":{\"events\":" << RD.nEvents << ",\"events_primary_entered_crystal\":" << RD.nPrimEntered << ",\"events_interacted\":" << RD.nInteracted << ",\"events_with_energy_deposit\":" << RD.nEdep
    << ",\"sum_energy_deposit_keV\":" << jn(RD.sumEdep) << ",\"max_abs_energy_closure_keV\":" << jn(RD.maxAbsClosure) << ",\"first_interaction_process_counts\":{";
  { bool f = true; for (auto& kv : RD.firstProc) { o << (f ? "" : ",") << js(kv.first) << ":" << kv.second; f = false; } }
  o << "}";
  if (g_full) {
    o << ",\"optical\":{\"generated_scintillation_photons\":" << RD.nScint << ",\"generated_wls_photons\":" << RD.nWLS << ",\"total_optical_tracks\":" << (RD.nScint + RD.nWLS)
      << ",\"photons_reaching_a_boundary\":" << RD.nBoundary << ",\"terminal\":{";
    for (int i = 0; i < T_N; ++i) o << (i ? "," : "") << js(kTermName[i]) << ":" << RD.term[i];
    o << "},\"rayleigh_scatters\":" << RD.nRay << ",\"mie_scatters\":" << RD.nMie << ",\"boundary_reflections\":" << RD.nRefl << ",\"photons_incident_on_sensor\":" << RD.nIncident << ",\"photons_detected\":" << RD.nDet
      << ",\"microcells_fired\":" << RD.cellsFired << ",\"photons_lost_to_saturation\":" << RD.cellsLost << ",\"total_charge_electrons\":" << jn(RD.charge)
      << ",\"face_loss\":{\"faces\":[\"+x\",\"-x\",\"+y\",\"-y\",\"+z\",\"-z\",\"coupling_edge\"],\"surface_absorbed\":[";
    for (int i = 0; i < 7; ++i) o << (i ? "," : "") << RD.faceLoss[0][i];
    o << "],\"escaped\":["; for (int i = 0; i < 7; ++i) o << (i ? "," : "") << RD.faceLoss[1][i];
    o << "]}}";
  }
  o << "},\n";
  o << "\"gamma_calculations\":{\"source\":\"Geant4 G4EmCalculator cross-section tables (calculated, not simulated)\",\"total_attenuation_length_mm\":" << jn(RD.attLen_mm) << ",\"mu_total_per_mm\":" << (RD.attLen_mm > 0 ? jn(1 / RD.attLen_mm) : "null")
    << ",\"mu_photoelectric_per_mm\":" << jn(RD.muPhot) << ",\"mu_compton_per_mm\":" << jn(RD.muCompt) << ",\"mu_pair_per_mm\":" << jn(RD.muConv) << ",\"mu_rayleigh_per_mm\":" << jn(RD.muRayl)
    << ",\"pencil_beam_chord_length_mm\":" << (RD.chord_mm >= 0 ? jn(RD.chord_mm) : "null") << ",\"expected_interaction_probability_pencil_beam\":" << (RD.expectedPint >= 0 ? jn(RD.expectedPint) : "null") << "},\n";
  o << "\"histograms\":{";
  if (g_full) o << "\"generated_wavelength_nm\":" << jhist(RD.hWl) << ",\"emission_time_ns\":" << jhist(RD.hTemit) << ",\"sensor_arrival_time_ns\":" << jhist(RD.hTarr) << ",\"detection_time_ns\":" << jhist(RD.hTdet) << ",";
  o << "\"edep_map_xy_keV\":" << jhist2(RD.mapXY) << ",\"edep_map_xz_keV\":" << jhist2(RD.mapXZ) << ",\"edep_map_yz_keV\":" << jhist2(RD.mapYZ) << "},\n";
  o << "\"time_reference\":\"all times are relative to the first energy deposit in the crystal of each event\",\n";
  o << "\"photon_log\":{\"mode\":" << js(C.photonLog) << ",\"rows_written\":" << RD.phRows << ",\"truncated\":" << (RD.phTruncated ? "true" : "false") << "},\n";
  o << "\"config_yaml_text\":" << js(g_configText) << ",\n\"config\":" << yamlToJson(g_root) << "\n}\n";
  std::ofstream f(path); f << o.str();
}

// ============================================================================
// 6. MAIN
// ============================================================================
static void PrintReport() {
  for (char s : {'E', 'W', 'I'})
    for (auto& i : g_issues) if (i.sev == s) std::cout << (s == 'E' ? "  [ERROR]   " : s == 'W' ? "  [warning] " : "  [info]    ") << i.path << ": " << i.msg << "\n";
  std::cout << "Stage readiness:\n";
  for (const char* nm : {"gamma_screening", "scintillation", "optical_transport", "photodetection"}) {
    const StageInfo& s = g_stage[nm];
    std::cout << "  " << std::left << std::setw(18) << nm << (s.ok ? "inputs complete" : "CANNOT RUN - missing/unknown inputs:") << "\n";
    for (auto& m : s.missing) std::cout << "      - " << m << "\n";
  }
  int estimated = 0, assumed = 0; for (auto& p : g_prov) if (p.known) { estimated += p.status == "estimated"; assumed += p.status == "assumed"; }
  if (assumed) std::cout << "  NOTE: " << assumed << " input(s) have status 'assumed' - modelling inputs, not measured specifications.\n";
  if (estimated) std::cout << "  NOTE: " << estimated << " input(s) have status 'estimated' - results are conditional on them (listed in the report).\n";
}

int main(int argc, char** argv) {
  std::string cfgPath = "config.yaml", visMacroOverride; bool validateOnly = false, vis = false; long evOver = -1, seedOver = -1; std::string modeOver, outOver;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--validate") validateOnly = true;
    else if (a == "--vis") vis = true;
    else if (a == "--vis-macro" && i + 1 < argc) visMacroOverride = argv[++i];
    else if (a == "--events" && i + 1 < argc) evOver = std::atol(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seedOver = std::atol(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) modeOver = argv[++i];
    else if (a == "--outdir" && i + 1 < argc) outOver = argv[++i];
    else if (a == "-h" || a == "--help") { std::cout << "usage: CsPbBr3Sim [config.yaml] [--validate] [--events N] [--seed N] [--mode auto|screening|full] [--outdir DIR] [--vis | --vis-macro FILE]\n"; return 0; }
    else if (a.rfind("--", 0) != 0) cfgPath = a;
    else { std::cerr << "unknown option " << a << "\n"; return 2; }
  }
  std::cout << SOFTWARE_NAME << " v" << SOFTWARE_VERSION << "  (Geant4 " << G4VERSION_NUMBER << ")\nConfiguration: " << cfgPath << "\n";
  G4int prevLevel = 0; (void)prevLevel;
  bool ok = LoadConfig(cfgPath);
  if (evOver > 0) { C.events = evOver; C.cliOverrides.push_back({"events", std::to_string(evOver)}); }
  if (seedOver >= 0) { C.seed = seedOver; C.cliOverrides.push_back({"seed", std::to_string(seedOver)}); }
  if (!modeOver.empty()) { if (modeOver != "auto" && modeOver != "screening" && modeOver != "full") { std::cerr << "bad --mode\n"; return 2; } C.modeRequested = modeOver; C.cliOverrides.push_back({"mode", modeOver}); }
  if (!outOver.empty()) { C.outDir = outOver; C.cliOverrides.push_back({"outdir", outOver}); }
  PrintReport();
  if (!ok) { std::cerr << "\nConfiguration INVALID - nothing was run.\n"; return 2; }
  bool modeOk = DecideMode();
  std::cout << "Requested mode: " << C.modeRequested << "  ->  effective mode: " << C.modeEffective << "\n";
  if (!modeOk) { std::cerr << "\nmode: full was requested but required inputs are missing (listed above). Nothing was run.\n"; return 3; }
  if (C.modeRequested == "auto" && C.modeEffective == "gamma_screening")
    std::cout << "  Scintillation/optical/photodetection stages will NOT run; no scintillation performance will be claimed.\n";
  g_full = (C.modeEffective == "full");
  if (validateOnly) { std::cout << "Validation finished (no simulation run).\n"; return 0; }

  std::filesystem::create_directories(C.outDir + "/plots");
  RD.evOut.open(C.outDir + "/events.csv"); RD.evOut << std::setprecision(9);
  RD.evOut << "event_id,gamma_energy_keV,energy_entering_crystal_keV,energy_deposited_keV,energy_escaped_net_keV,energy_closure_keV,interacted,first_interaction_process,edep_centroid_x_mm,edep_centroid_y_mm,edep_centroid_z_mm,first_int_x_mm,first_int_y_mm,first_int_z_mm";
  if (g_full) RD.evOut << ",n_scint_photons,n_wls_photons,n_photons_reached_boundary,n_bulk_absorbed,n_surface_absorbed,n_escaped,n_wls_converted,n_unclassified,n_sensor_incident,n_detected_photoelectrons,n_sensor_not_detected,n_rayleigh_scatters,n_mie_scatters,n_boundary_reflections,n_cells_fired_or_signal,n_photons_lost_saturation,charge_electrons,t_first_detection_ns,t_mean_detection_ns";
  RD.evOut << "\n";
  RD.phOut.open(C.outDir + "/photon_data.csv"); RD.phOut << std::setprecision(7);
  RD.phOut << "event_id,track_id,parent_id,creator,wavelength_nm,t_emit_ns,x_emit_mm,y_emit_mm,z_emit_mm,terminal_class,t_end_ns,x_end_mm,y_end_mm,z_end_mm,face,n_boundary_interactions,n_reflections,n_rayleigh,n_mie,path_length_mm,detected\n";
  if (!g_full) C.photonLog = "none";

  // histogram ranges
  if (g_full) {
    std::vector<double> wl, I; std::string b; emissionPoints(wl, I, b);
    double lo = std::max(1.0, *std::min_element(wl.begin(), wl.end()) - 20), hi = *std::max_element(wl.begin(), wl.end()) + 20;
    RD.hWl.init(lo, hi, 100);
    double tauMax = 0; for (auto& c : C.comps) tauMax = std::max(tauMax, c.tau.value);
    double rise = C.riseTime.known ? C.riseTime.value : 0;
    double tEm = 10 * tauMax + 10 * rise + 1.0;
    double nmax = C.nCrystal.hasTable ? *std::max_element(C.nCrystal.ty.begin(), C.nCrystal.ty.end()) : C.nCrystal.value;
    double Lmax = std::sqrt(C.dx * C.dx + C.dy * C.dy + C.dz * C.dz);
    double transit = std::max(20.0, 50.0 * nmax * Lmax / 299.792458);
    RD.hTemit.init(0, tEm, 100); RD.hTarr.init(0, tEm + transit, 150); RD.hTdet.init(0, tEm + transit, 150);
  }
  RD.mapXY.init(-C.dx / 2, C.dx / 2, -C.dy / 2, C.dy / 2, 50, 50); RD.mapXZ.init(-C.dx / 2, C.dx / 2, -C.dz / 2, C.dz / 2, 50, 50); RD.mapYZ.init(-C.dy / 2, C.dy / 2, -C.dz / 2, C.dz / 2, 50, 50);

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
  G4Random::setTheSeed(C.seed);
  extendSpan(C.nCrystal); extendSpan(C.absLen); extendSpan(C.rayLen); extendSpan(C.mieLen); extendSpan(C.pde); extendSpan(C.surrN); extendSpan(C.winN); extendSpan(C.coupN); extendSpan(C.reflectivity);
  rm->SetUserInitialization(new Detector);
  rm->SetUserInitialization(BuildPhysics());
  rm->SetUserInitialization(new Actions);
  rm->Initialize();
  auto* ui = G4UImanager::GetUIpointer();
  ui->ApplyCommand("/run/verbose 0"); ui->ApplyCommand("/event/verbose 0"); ui->ApplyCommand("/tracking/verbose 0");
  ui->ApplyCommand("/run/printProgress " + std::to_string(std::max(1L, C.events / 10)));
  std::cout << "Running " << C.events << " events (" << C.modeEffective << ", seed " << C.seed << ") ...\n";
  RD.timestamp = nowIso();
  auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<G4UImanager, void (*)(G4UImanager*)> dummy(nullptr, [](G4UImanager*) {});
#ifdef HAVE_G4VIS
  std::unique_ptr<G4VisExecutive> visMgr; std::unique_ptr<G4UIExecutive> session;
  if (vis || !visMacroOverride.empty()) {
    // The UI session must exist BEFORE the viewer is opened, otherwise /vis/open OGL creates a
    // separate stand-alone window instead of the viewer embedded in the Qt interface.
    if (visMacroOverride.empty()) { int ac = 1; char* av[] = {argv[0]}; session.reset(new G4UIExecutive(ac, av)); }
    visMgr.reset(new G4VisExecutive("quiet")); visMgr->Initialize();
    std::string m = visMacroOverride.empty() ? C.visMacro : visMacroOverride;
    ui->ApplyCommand("/control/execute " + m);
  }
#else
  if (vis || !visMacroOverride.empty()) std::cerr << "WARNING: built without Geant4 visualisation; ignoring --vis\n";
#endif
  rm->BeamOn((G4int)C.events);
#ifdef HAVE_G4VIS
  if (vis || !visMacroOverride.empty()) {
    // End-of-event accumulation defers drawing. Explicitly flush after BeamOn
    // so the kept trajectories are rendered before entering the UI session.
    ui->ApplyCommand("/vis/enable");
    ui->ApplyCommand("/vis/viewer/flush");
  }
  if (session) { std::cout << "Interactive session (type 'exit' to finish).\n"; session->SessionStart(); }
#endif
  RD.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ComputeGammaTables();
  RD.evOut.close(); RD.phOut.close();
  WriteSummary(C.outDir + "/summary.json");
  std::cout << "Done in " << RD.wall_s << " s.  Output: " << C.outDir << "/{events.csv,photon_data.csv,summary.json}\n";
  if (g_full) std::cout << "  generated scintillation photons: " << RD.nScint << ", reached sensor: " << RD.nIncident << ", detected: " << RD.nDet << "\n";
  std::cout << "  events interacting: " << RD.nInteracted << "/" << RD.nEvents << ", max |energy closure| = " << RD.maxAbsClosure << " keV\n";
  delete rm;
  return 0;
}