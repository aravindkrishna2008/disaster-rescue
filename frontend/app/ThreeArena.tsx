'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Obstacle {
  pos: number[];
  size: number[];
}

interface Hazard {
  center: number[];
  radius: number;
}

interface ThreeArenaProps {
  robotPos: { x: number; y: number };
  robotHeading: number;
  activeTarget: 'CHILD' | 'ADULT' | null;
  trajectory: number[][] | null;
  obstacles: Obstacle[];
  hazards: Hazard[];
  survivors: {
    CHILD: { x: number; y: number };
    ADULT: { x: number; y: number };
  };
}

export default function ThreeArena({
  robotPos,
  robotHeading,
  activeTarget,
  trajectory,
  obstacles,
  hazards,
  survivors,
}: ThreeArenaProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const robotRef = useRef<THREE.Group | null>(null);
  const pathLineRef = useRef<THREE.Line | null>(null);
  const targetBeaconsRef = useRef<{ CHILD: THREE.Group; ADULT: THREE.Group } | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);

  // Map MuJoCo coordinates to Three.js coordinates
  // MuJoCo: X right, Y forward, Z up
  // Three.js: X right, Y up, Z backward (-Z forward)
  // Mapping:
  // 3D X = MuJoCo X
  // 3D Y = MuJoCo Z (height)
  // 3D Z = -MuJoCo Y
  const to3D = (x: number, y: number, z = 0) => {
    return new THREE.Vector3(x, z, -y);
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f0ede4'); // matches --bg
    sceneRef.current = scene;

    // Add some subtle grid fog
    scene.fog = new THREE.FogExp2('#f0ede4', 0.04);

    // 2. Camera setup
    const width = container.clientWidth;
    const height = container.clientHeight || 450;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 12, 16); // Cinematic high angle

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
    controls.minDistance = 3;
    controls.maxDistance = 35;
    controls.target.set(0, 0, 0);

    // 5. Lights
    const ambientLight = new THREE.AmbientLight('#ffffff', 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight('#ffffff', 0.8);
    dirLight.position.set(10, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.bias = -0.001;
    scene.add(dirLight);

    // Add a subtle blue/cyan point light from the bottom for technical feel
    const pointLight = new THREE.PointLight('#00a8ff', 0.5, 30);
    pointLight.position.set(0, -2, 0);
    scene.add(pointLight);

    // 6. Ground Grid (Tactical Paper style)
    const gridHelper = new THREE.GridHelper(20, 20, '#14110e', '#c2bbac');
    gridHelper.position.y = -0.01; // Slightly below floor to prevent z-fighting
    scene.add(gridHelper);

    // Grid border
    const borderGeom = new THREE.BoxGeometry(20, 0.05, 20);
    const borderMat = new THREE.MeshBasicMaterial({
      color: '#d8d2c4',
      wireframe: true,
    });
    const border = new THREE.Mesh(borderGeom, borderMat);
    border.position.y = -0.025;
    scene.add(border);

    // 7. Add Obstacles (from DEFAULT_SCENE)
    const obstacleGeometries: THREE.BoxGeometry[] = [];
    const obstacleMaterial = new THREE.MeshStandardMaterial({
      color: '#b02e26', // matches --red
      transparent: true,
      opacity: 0.18,
      roughness: 0.2,
      metalness: 0.1,
    });
    const obstacleWireframeMaterial = new THREE.MeshBasicMaterial({
      color: '#b02e26',
      wireframe: true,
    });

    obstacles.forEach((obs, idx) => {
      // MuJoCo size is half-extents. Three.js BoxGeometry is full sizes.
      const sizeX = obs.size[0] * 2;
      const sizeY = obs.size[2] * 2; // MuJoCo Z is height
      const sizeZ = obs.size[1] * 2; // MuJoCo Y is width
      
      const geom = new THREE.BoxGeometry(sizeX, sizeY, sizeZ);
      obstacleGeometries.push(geom);

      const mesh = new THREE.Mesh(geom, obstacleMaterial);
      // Position: MuJoCo: [x, y, z] -> Three.js: [x, z_mujoco, -y_mujoco]
      mesh.position.set(obs.pos[0], obs.pos[2], -obs.pos[1]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);

      // Add wireframe outline
      const wireframe = new THREE.Mesh(geom, obstacleWireframeMaterial);
      wireframe.position.copy(mesh.position);
      scene.add(wireframe);
    });

    // 8. Add Hazard Zones (circles on the floor)
    hazards.forEach((hazard) => {
      const radius = hazard.radius;
      // We can use a RingGeometry or a TorusGeometry lying flat on the floor
      const torusGeom = new THREE.TorusGeometry(radius, 0.04, 8, 48);
      const torusMat = new THREE.MeshBasicMaterial({
        color: '#b02e26',
        transparent: true,
        opacity: 0.5,
      });
      const torus = new THREE.Mesh(torusGeom, torusMat);
      torus.rotation.x = Math.PI / 2; // lay flat
      torus.position.set(hazard.center[0], 0.01, -hazard.center[1]);
      scene.add(torus);

      // Radial pattern inside
      const discGeom = new THREE.RingGeometry(0, radius, 32);
      const discMat = new THREE.MeshBasicMaterial({
        color: '#b02e26',
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
      });
      const disc = new THREE.Mesh(discGeom, discMat);
      disc.rotation.x = Math.PI / 2;
      disc.position.copy(torus.position);
      scene.add(disc);
    });

    // 9. Add Survivors (Child and Adult Beacons)
    const createSurvivorBeacon = (color: string, label: string, pos: { x: number; y: number }) => {
      const group = new THREE.Group();
      const coords = to3D(pos.x, pos.y, 0.25);
      group.position.copy(coords);

      // Glowing orb
      const orbGeom = new THREE.SphereGeometry(0.2, 16, 16);
      const orbMat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 1.5,
        roughness: 0.1,
      });
      const orb = new THREE.Mesh(orbGeom, orbMat);
      group.add(orb);

      // Cylinder beacon beam
      const beamGeom = new THREE.CylinderGeometry(0.15, 0.15, 3.0, 16, 1, true);
      const beamMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
      });
      const beam = new THREE.Mesh(beamGeom, beamMat);
      beam.position.y = 1.5; // Offset cylinder center
      group.add(beam);

      // Glowing pulse rings on floor
      const rings: THREE.Mesh[] = [];
      const ringGeom = new THREE.RingGeometry(0.1, 0.7, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      
      const pulseRing = new THREE.Mesh(ringGeom, ringMat);
      pulseRing.rotation.x = Math.PI / 2;
      pulseRing.position.y = -0.24; // flat on floor
      group.add(pulseRing);
      rings.push(pulseRing);

      scene.add(group);
      return { group, rings, color };
    };

    const childBeacon = createSurvivorBeacon('#00d2ff', 'CHILD', survivors.CHILD); // cyan
    const adultBeacon = createSurvivorBeacon('#ff9f43', 'ADULT', survivors.ADULT); // orange

    targetBeaconsRef.current = {
      CHILD: childBeacon.group,
      ADULT: adultBeacon.group,
    };

    // 10. Add Robot model (sleek mechanical cone/mesh)
    const robotGroup = new THREE.Group();
    robotGroup.position.copy(to3D(robotPos.x, robotPos.y, 0.35));
    
    // Robot body (sleek metallic sphere)
    const bodyGeom = new THREE.SphereGeometry(0.3, 16, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: '#14110e',
      roughness: 0.1,
      metalness: 0.9,
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    body.castShadow = true;
    robotGroup.add(body);

    // Glowing core ring
    const coreGeom = new THREE.CylinderGeometry(0.31, 0.31, 0.08, 16);
    const coreMat = new THREE.MeshBasicMaterial({ color: '#b02e26' }); // default red
    const core = new THREE.Mesh(coreGeom, coreMat);
    robotGroup.add(core);

    // Heading indicator (arrow pointing forward)
    const arrowGeom = new THREE.ConeGeometry(0.12, 0.4, 4);
    const arrowMat = new THREE.MeshBasicMaterial({ color: '#14110e' });
    const arrow = new THREE.Mesh(arrowGeom, arrowMat);
    arrow.rotation.x = -Math.PI / 2; // point forward along -Z in local space
    arrow.position.set(0, 0, -0.45);
    robotGroup.add(arrow);

    // Heading glow
    const headGlowGeom = new THREE.SphereGeometry(0.08, 8, 8);
    const headGlowMat = new THREE.MeshBasicMaterial({ color: '#ff4d4d' });
    const headGlow = new THREE.Mesh(headGlowGeom, headGlowMat);
    headGlow.position.set(0, 0, -0.32);
    robotGroup.add(headGlow);

    scene.add(robotGroup);
    robotRef.current = robotGroup;

    // 11. Trajectory path line
    const pathMat = new THREE.LineBasicMaterial({
      color: '#b02e26', // matches --red
      linewidth: 3, // note: linewidth > 1 usually ignored by WebGL implementations, but looks fine
    });
    const pathGeom = new THREE.BufferGeometry();
    const pathLine = new THREE.Line(pathGeom, pathMat);
    scene.add(pathLine);
    pathLineRef.current = pathLine;

    // 12. Animation loop
    let animationFrameId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const elapsed = clock.getElapsedTime();

      // Animate survivor rings pulsing
      [childBeacon, adultBeacon].forEach((b) => {
        b.rings.forEach((ring) => {
          const scale = 1 + (elapsed % 1.5) * 1.8;
          ring.scale.set(scale, scale, 1);
          
          const mat = ring.material as THREE.MeshBasicMaterial;
          mat.opacity = 0.4 * (1 - (elapsed % 1.5) / 1.5);
        });

        // Make the orb float up and down slightly
        b.group.children[0].position.y = Math.sin(elapsed * 4) * 0.08;
      });

      // Orbit controls update
      controls.update();

      // Render
      renderer.render(scene, camera);
    };
    animate();

    // 13. Window resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 450;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      container.removeChild(renderer.domElement);
      renderer.dispose();
      
      // dispose geometries/materials
      scene.traverse((object: any) => {
        if (object.geometry) {
          object.geometry.dispose();
        }
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach((mat) => mat.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
    };
  }, [obstacles, hazards, survivors]);

  // Update robot position & heading
  useEffect(() => {
    if (robotRef.current) {
      const pos3D = to3D(robotPos.x, robotPos.y, 0.3);
      robotRef.current.position.copy(pos3D);
      
      // heading is in degrees, counterclockwise in MuJoCo, where 0 is east (+X).
      // In Three.js, rotation is around Y axis, counterclockwise, where 0 is south (+Z) or north (-Z).
      // We map heading so the pointer faces the correct direction of travel.
      // MuJoCo heading: 0 -> pointing +X (East).
      // Three.js angle: pointing toward -Z (North) is 0.
      // So to point East (+X), we rotate -Math.PI / 2.
      // Formula: rad = (heading * Math.PI) / 180
      // Let's set rotation.y = -rad + Math.PI / 2
      const rad = (robotHeading * Math.PI) / 180;
      robotRef.current.rotation.y = -rad + Math.PI / 2;
    }
  }, [robotPos, robotHeading]);

  // Update target beacon styling (make the active target glow more intensely)
  useEffect(() => {
    if (targetBeaconsRef.current) {
      const childGroup = targetBeaconsRef.current.CHILD;
      const adultGroup = targetBeaconsRef.current.ADULT;

      const setIntensity = (group: THREE.Group, isActive: boolean) => {
        // orb is child 0, beam is child 1
        const orb = group.children[0] as THREE.Mesh;
        const beam = group.children[1] as THREE.Mesh;

        if (orb && orb.material) {
          const mat = orb.material as THREE.MeshStandardMaterial;
          mat.emissiveIntensity = isActive ? 3.0 : 0.8;
          orb.scale.setScalar(isActive ? 1.4 : 1.0);
        }
        if (beam && beam.material) {
          const mat = beam.material as THREE.MeshBasicMaterial;
          mat.opacity = isActive ? 0.6 : 0.15;
          beam.scale.set(isActive ? 1.5 : 1.0, 1.0, isActive ? 1.5 : 1.0);
        }
      };

      setIntensity(childGroup, activeTarget === 'CHILD');
      setIntensity(adultGroup, activeTarget === 'ADULT');
    }
  }, [activeTarget]);

  // Update trajectory path line
  useEffect(() => {
    if (pathLineRef.current && trajectory && trajectory.length > 0) {
      const points = trajectory.map((pt) => to3D(pt[0], pt[1], 0.02)); // slightly above grid
      const pathGeom = pathLineRef.current.geometry;
      pathGeom.setFromPoints(points);
      pathLineRef.current.visible = true;
    } else if (pathLineRef.current) {
      pathLineRef.current.visible = false;
    }
  }, [trajectory]);

  return (
    <div
      ref={mountRef}
      style={{
        width: '100%',
        height: '100%',
        minHeight: '400px',
        position: 'relative',
        borderRadius: '1px',
        border: '1px solid var(--rule)',
        overflow: 'hidden',
      }}
    />
  );
}
