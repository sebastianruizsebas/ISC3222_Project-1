% INDEX - File Navigation Guide
%
% This file helps you find what you need
%
% Run this file to see all available resources and their purposes

clear; clc;

fprintf('\n');
fprintf('╔═════════════════════════════════════════════════════════════╗\n');
fprintf('║     INTERCEPTION GAME - FILE INDEX & NAVIGATION GUIDE       ║\n');
fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

fprintf('📖 DOCUMENTATION (Read These First)\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('START HERE:\n');
fprintf('  GETTING_STARTED.md\n');
fprintf('    → Final summary + quick start guide\n');
fprintf('    → Read this first! (5 minutes)\n\n');

fprintf('THEN READ:\n');
fprintf('  README.md\n');
fprintf('    → Architecture overview\n');
fprintf('    → Full API reference\n');
fprintf('    → Neural interpretation guide\n');
fprintf('    → How to extend code\n\n');

fprintf('SETUP & TROUBLESHOOTING:\n');
fprintf('  SETUP.md\n');
fprintf('    → Installation instructions\n');
fprintf('    → Troubleshooting common issues\n');
fprintf('    → Customization guide\n');
fprintf('    → Batch processing examples\n\n');

fprintf('DETAILED CHANGES:\n');
fprintf('  REFACTORING_SUMMARY.md\n');
fprintf('    → What changed from original code\n');
fprintf('    → Why it was refactored\n');
fprintf('    → Benefits of new design\n');
fprintf('    → Migration guide\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('💻 SOURCE CODE (Main Classes)\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('GAME LOGIC (+Game/ package):\n');
fprintf('  GameConfiguration.m\n');
fprintf('    → Manages experiment parameters\n');
fprintf('    → Validates configuration\n');
fprintf('    → Saves/loads from JSON\n');
fprintf('    → Usage: config = Game.GameConfiguration();\n\n');

fprintf('  GameEngine.m\n');
fprintf('    → Main game loop\n');
fprintf('    → Graphics rendering\n');
fprintf('    → Keyboard/mouse input\n');
fprintf('    → Trial execution\n');
fprintf('    → Usage: engine = Game.GameEngine(config); engine.run();\n\n');

fprintf('  TrialData.m\n');
fprintf('    → Container for single trial results\n');
fprintf('    → Stores motor commands, accuracy, etc.\n');
fprintf('    → Computes derived metrics\n');
fprintf('    → Usage: trial.addMotorCommand(t, pos);\n\n');

fprintf('ANALYSIS (+Analysis/ package):\n');
fprintf('  ExperimentManager.m\n');
fprintf('    → Orchestrates full experiment workflow\n');
fprintf('    → Handles participant intake\n');
fprintf('    → Generates reports\n');
fprintf('    → Saves results\n');
fprintf('    → Usage: exp = Analysis.ExperimentManager(); exp.runFull();\n\n');

fprintf('MODELS (+Models/ package):\n');
fprintf('  HierarchicalMotionModel.m\n');
fprintf('    → Fits hierarchical inference model\n');
fprintf('    → Estimates precision weights (π_x, π_v, π_a)\n');
fprintf('    → Provides neural interpretation\n');
fprintf('    → Usage: model = Models.HierarchicalMotionModel(trial); model.fitPrecision();\n\n');

fprintf('UTILITIES (+Utils/ package):\n');
fprintf('  TrajectoryGenerator.m\n');
fprintf('    → Generates target motion stimuli\n');
fprintf('    → Supports: constant, accelerating, decelerating motion\n');
fprintf('    → Usage: [traj, times, v, a] = Utils.TrajectoryGenerator.generateTargetTrajectory(...);\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('🚀 ENTRY POINTS (Run These)\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('  run_experiment.m\n');
fprintf('    → Main entry point for full experiment\n');
fprintf('    → Run: run_experiment()\n');
fprintf('    → Does: participant intake → game → analysis → report\n\n');

fprintf('  test_all.m\n');
fprintf('    → Unit tests for all components\n');
fprintf('    → Run: test_all()\n');
fprintf('    → Tests: configuration, trial data, trajectory gen, model fitting\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('📚 EXAMPLES & REFERENCE\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('  EXAMPLES.m\n');
fprintf('    → 8 complete usage patterns\n');
fprintf('    → From simple to advanced\n');
fprintf('    → All code copy-paste ready\n');
fprintf('    → Read/run: EXAMPLES\n\n');

fprintf('  QUICK_REFERENCE.m\n');
fprintf('    → All classes and methods at a glance\n');
fprintf('    → Common one-liners\n');
fprintf('    → Field names and types\n');
fprintf('    → Useful plotting code\n');
fprintf('    → Keep open while coding\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('📂 DIRECTORY STRUCTURE\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('InterceptionGame/\n');
fprintf('├── +Game/                    CORE GAME LOGIC\n');
fprintf('│   ├── GameConfiguration.m   - Parameter management\n');
fprintf('│   ├── GameEngine.m          - Game loop & graphics\n');
fprintf('│   └── TrialData.m           - Trial data container\n');
fprintf('│\n');
fprintf('├── +Analysis/                DATA ANALYSIS\n');
fprintf('│   └── ExperimentManager.m   - Workflow orchestration\n');
fprintf('│\n');
fprintf('├── +Models/                  STATISTICAL MODELS\n');
fprintf('│   └── HierarchicalMotionModel.m - Precision fitting\n');
fprintf('│\n');
fprintf('├── +Utils/                   UTILITIES\n');
fprintf('│   └── TrajectoryGenerator.m - Motion stimulus generation\n');
fprintf('│\n');
fprintf('├── tests/\n');
fprintf('│   └── test_all.m            - Unit tests\n');
fprintf('│\n');
fprintf('├── data/                     OUTPUT DIRECTORY (auto-created)\n');
fprintf('│\n');
fprintf('├── run_experiment.m          - Main entry point\n');
fprintf('├── EXAMPLES.m                - Usage patterns\n');
fprintf('├── QUICK_REFERENCE.m         - API quick lookup\n');
fprintf('├── INDEX.m                   - This file\n');
fprintf('│\n');
fprintf('├── README.md                 - Full documentation\n');
fprintf('├── SETUP.md                  - Installation & troubleshooting\n');
fprintf('├── GETTING_STARTED.md        - Quick start guide\n');
fprintf('└── REFACTORING_SUMMARY.md    - What changed & why\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('✅ QUICK START CHECKLIST\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('[ ] 1. Read GETTING_STARTED.md (5 min)\n');
fprintf('[ ] 2. Run: test_all() (verify setup)\n');
fprintf('[ ] 3. Run: run_experiment() (test game)\n');
fprintf('[ ] 4. Review output in interception_game_results/\n');
fprintf('[ ] 5. Read: README.md (detailed docs)\n');
fprintf('[ ] 6. Run: EXAMPLES (see usage patterns)\n');
fprintf('[ ] 7. Start your experiment!\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('🎯 COMMON TASKS\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('\"I want to run the game\"\n');
fprintf('  → run_experiment()\n\n');

fprintf('\"I want to customize the game\"\n');
fprintf('  → Read: README.md section \"Class Reference\"\n');
fprintf('  → Or: EXAMPLES.m example #2\n\n');

fprintf('\"I want to understand the code\"\n');
fprintf('  → Read: README.md \"Architecture\"\n');
fprintf('  → Then: Inline class documentation in +Game/\n\n');

fprintf('\"I want to analyze results\"\n');
fprintf('  → See: EXAMPLES.m examples #4, #5, #6\n');
fprintf('  → Or: QUICK_REFERENCE.m section \"ANALYZING EXISTING DATA\"\n\n');

fprintf('\"I want to add new features\"\n');
fprintf('  → Read: README.md \"Extending the Code\"\n');
fprintf('  → Follow: Existing class patterns\n');
fprintf('  → Add: Tests in tests/\n\n');

fprintf('\"Something is not working\"\n');
fprintf('  → Run: test_all() (check setup)\n');
fprintf('  → Read: SETUP.md \"Troubleshooting\"\n');
fprintf('  → Then: Try EXAMPLES.m\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('💡 WHERE TO FIND THINGS\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('Topic                       Location\n');
fprintf('────────────────────────────────────────────────────────────────\n');
fprintf('Architecture overview       README.md\n');
fprintf('Full API reference          README.md + inline docs\n');
fprintf('Quick start                 GETTING_STARTED.md\n');
fprintf('Installation help           SETUP.md\n');
fprintf('Troubleshooting             SETUP.md\n');
fprintf('Usage examples              EXAMPLES.m\n');
fprintf('Quick lookup API            QUICK_REFERENCE.m\n');
fprintf('Test components             test_all.m\n');
fprintf('Neural interpretation       README.md + HierarchicalMotionModel.m\n');
fprintf('Customization examples      SETUP.md\n');
fprintf('What changed                REFACTORING_SUMMARY.md\n');
fprintf('Main entry point            run_experiment.m\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('📞 HELP RESOURCES\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('Installation problems?\n');
fprintf('  → SETUP.md → Troubleshooting section\n\n');

fprintf('Not sure how to use classes?\n');
fprintf('  → QUICK_REFERENCE.m or run: EXAMPLES\n\n');

fprintf('Want to understand architecture?\n');
fprintf('  → README.md → Architecture section\n\n');

fprintf('Need to customize game difficulty?\n');
fprintf('  → SETUP.md → \"Common Customizations\"\n\n');

fprintf('Want to analyze your results?\n');
fprintf('  → EXAMPLES.m → Examples #4-6\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('✨ KEY DOCUMENTATION PAGES\n');
fprintf('───────────────────────────────────────────────────────────────\n\n');

fprintf('To view in MATLAB:\n');
fprintf('  edit README.md               → Full documentation\n');
fprintf('  edit SETUP.md                → Installation guide\n');
fprintf('  edit GETTING_STARTED.md      → Quick start\n');
fprintf('  edit EXAMPLES.m              → Code examples\n');
fprintf('  edit QUICK_REFERENCE.m       → API quick lookup\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');

fprintf('🎯 NEXT STEP: Read GETTING_STARTED.md\n\n');

fprintf('Then run:\n');
fprintf('  >> run_experiment()\n\n');

fprintf('═════════════════════════════════════════════════════════════\n\n');
