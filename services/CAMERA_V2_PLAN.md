# Camera System V2 - Implementation Plan

## Overview

This document outlines the plan for the rewritten camera and click_with_camera systems.

## Architecture

### Module Structure

```
services/
├── camera_v2.py              # Core camera control system
├── click_with_camera_v2.py    # Click system with camera integration
└── CAMERA_V2_PLAN.md          # This file
```

## Camera System V2 (`camera_v2.py`)

### Components

1. **CameraStateManager**
   - Manages camera state (IDLE, TARGETING, FOLLOWING, LOCKED)
   - Stores configuration
   - Thread-safe state access

2. **CameraMovementExecutor**
   - Executes camera movements via IPC
   - Thread-based movement queue
   - Handles yaw, pitch, zoom movements
   - Waits for camera stability

3. **CameraTargeting**
   - Calculates camera movements to position targets
   - Handles iterative refinement
   - Manages screen position calculations

4. **CameraController**
   - Main interface for camera operations
   - Coordinates all components
   - Provides high-level API

### Key Features

- Clean separation of concerns
- Thread-safe operations
- Iterative refinement for large movements
- Configurable movement parameters
- Stability detection

## Click with Camera System V2 (`click_with_camera_v2.py`)

### Components

1. **ScreenCoordinateManager**
   - Gets screen coordinates for objects/NPCs/ground
   - Calculates optimal click points
   - Handles UI blocking detection

2. **InteractionVerifier**
   - Verifies object interactions
   - Verifies NPC interactions
   - Verifies ground clicks

3. **ClickWithCameraHandler**
   - Main handler for click operations
   - Coordinates camera movement and clicking
   - Handles retries and error recovery

### Key Features

- Unified interface for all click types
- Automatic camera positioning
- Interaction verification
- Retry logic with camera adjustments
- UI blocking detection and avoidance

## Implementation Steps

### Phase 1: Core Camera System
1. Implement CameraStateManager
2. Implement CameraMovementExecutor (thread system)
3. Implement basic movement execution (yaw, pitch, zoom)
4. Implement stability detection

### Phase 2: Camera Targeting
1. Implement screen position lookup
2. Implement movement calculation
3. Implement iterative refinement
4. Add error handling and retries

### Phase 3: Click System
1. Implement ScreenCoordinateManager
2. Implement InteractionVerifier
3. Implement ClickWithCameraHandler
4. Add convenience functions

### Phase 4: Integration
1. Replace old system calls with new system
2. Update all call sites
3. Add comprehensive error handling
4. Performance optimization

## Migration Strategy

1. Keep old system (`camera_integration.py`, `click_with_camera.py`) intact
2. Implement new system alongside old system
3. Add feature flags to switch between systems
4. Gradually migrate call sites
5. Remove old system once migration complete

## TODO

- [ ] Implement CameraStateManager
- [ ] Implement CameraMovementExecutor thread system
- [ ] Implement basic camera movements (yaw, pitch, zoom)
- [ ] Implement stability detection
- [ ] Implement screen position lookup
- [ ] Implement movement calculation
- [ ] Implement iterative refinement
- [ ] Implement ScreenCoordinateManager
- [ ] Implement InteractionVerifier
- [ ] Implement ClickWithCameraHandler
- [ ] Add comprehensive error handling
- [ ] Add logging and debugging
- [ ] Write tests
- [ ] Update documentation
