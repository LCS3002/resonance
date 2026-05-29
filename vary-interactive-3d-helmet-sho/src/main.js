import './styles/tokens.css';
import './styles/base.css';
import './styles/animations.css';
import './styles/atoms.css';
import './styles/molecules.css';
import './styles/organisms.css';

import { initBrainCanvas } from './organisms/BrainCanvas.js';
import { initDemoOverlay }  from './organisms/DemoOverlay.js';
import { initStudioOverlay } from './organisms/StudioOverlay.js';

initBrainCanvas(
  document.getElementById('brainCanvas'),
  document.getElementById('modeToggle')
);

initDemoOverlay();
initStudioOverlay();
