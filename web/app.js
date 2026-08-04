// Loaded on demand inside enableCamera() rather than as a static top-level import:
// a CDN failure must not stop the module from running, or mouse mode dies with it.
const TASKS_VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

// Tune these values if a particular webcam or lighting setup needs adjustment.
const MIN_DRAW_MS = 120, COOLDOWN_MS = 450, EMA_ALPHA = 0.30;
// The draw is driven purely by hand depth, so no fist is needed to hold or throw the
// ball. Finger closedness is the noisiest signal MediaPipe gives us, so nothing
// required depends on it any more.
const BASELINE_ALPHA = 0.03, DRAW_ENTER = 0.07, DRAW_CANCEL = 0.02;
const MAX_DEPTH_PULL = 0.28, MAX_RELEASE_SPEED = 1.6, THRUST_TRIGGER = 0.34;
const PULL_WEIGHT = 0.70, MIN_LAUNCH_POWER = 0.18;
// A slow forward move fires too: releasing this much of the draw counts as a throw.
const MIN_CHARGED_PULL = 0.18, RELEASE_FRACTION = 0.45;
// Optional convenience trigger: opening a hand that happened to be closed also fires.
const FIST_HELD_THRESHOLD = 0.70, OPEN_THRESHOLD = 0.48;
// MAX_SPEED is tuned so every target is reachable on draw distance alone: the hardest
// shot needs ~0.70 power against a draw-only ceiling of ~0.75.
const W = 1280, H = 720, GROUND = 650, GRAVITY = 900, MAX_SPEED = 1450, MAX_PULL = 152, RELOAD_DELAY = .7, LEVEL_CLEAR_DELAY = 1.5;
// Each level is a name and its targets.
const LEVELS = [
  {name:"WARM UP", targets:[{x:850,y:570,w:46,h:80,value:100},{x:970,y:530,w:54,h:120,value:150},{x:1090,y:590,w:62,h:60,value:100}]},
  {name:"LONG SHOT", targets:[{x:880,y:590,w:34,h:60,value:150},{x:985,y:500,w:38,h:150,value:200},{x:1100,y:570,w:40,h:80,value:200},{x:1200,y:380,w:34,h:60,value:250}]},
];
const PALM = [0, 5, 9, 13, 17], TIPS = [4, 8, 12, 16, 20];
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const canvas = document.querySelector("#game"), ctx = canvas.getContext("2d");
const video = document.querySelector("#camera"), overlay = document.querySelector("#hand-overlay"), handCtx = overlay.getContext("2d");
const cameraStatus = document.querySelector("#camera-status"), trackingStatus = document.querySelector("#tracking-status");
const cameraPicker = document.querySelector("#camera-picker"), cameraSelect = document.querySelector("#camera-select");
let landmarker, lastVideoTime = -1, gestureState = "idle", smoothHand, drawStarted = 0, cooldownUntil = 0;
let grabSize = 0, peakPull = 0, baseline = 0, wasClosed = false, lastSize = 0, lastHandTime = 0, mouseMode = false, mouseDown = false, mousePull = 0, mouseAim = {x:.5,y:.5};
let projectile, velocity, flying, restTimer = 0, score, shots, level = 0, completed = false, clearTimer = 0, targets, particles, lastFrame = performance.now();
let activeInput = {aim:{x:.5,y:.5},pull:0,drawing:false};
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);

function reset() { score=0; shots=0; completed=false; particles=[]; loadLevel(0); }
function loadLevel(index) {
  level=index; clearTimer=0;
  targets=LEVELS[index].targets.map(t=>({...t,alive:true}));
  reloadSling();
}
function updateLevels(dt) {
  // Advance a beat after the last target falls, so the hit reads before the swap.
  if(completed||targets.some(t=>t.alive)){clearTimer=0;return;}
  clearTimer+=dt;
  if(clearTimer<LEVEL_CLEAR_DELAY)return;
  if(level+1<LEVELS.length) loadLevel(level+1); else {completed=true;clearTimer=0;}
}
function direction(aim) {
  const radians = clamp(-55+(aim.x-.5)*46+(.5-aim.y)*34,-84,-18)*Math.PI/180;
  return {x:Math.cos(radians),y:Math.sin(radians)};
}
function analyze(landmarks) {
  const palm = PALM.reduce((sum,index) => ({x:sum.x+landmarks[index].x/PALM.length,y:sum.y+landmarks[index].y/PALM.length}),{x:0,y:0});
  const size = Math.max(distance(landmarks[0],landmarks[9]),.0001);
  const tipDistance = TIPS.reduce((sum,index)=>sum+distance(landmarks[index],palm),0)/TIPS.length/size;
  const closedness = clamp((1.45-tipDistance)/(1.45-.62));
  if (!smoothHand) smoothHand={palm,size,closedness};
  else { smoothHand.palm={x:smoothHand.palm.x+(palm.x-smoothHand.palm.x)*EMA_ALPHA,y:smoothHand.palm.y+(palm.y-smoothHand.palm.y)*EMA_ALPHA}; smoothHand.size+= (size-smoothHand.size)*EMA_ALPHA; smoothHand.closedness+=(closedness-smoothHand.closedness)*EMA_ALPHA; }
  return smoothHand;
}
function control(hand, now) {
  // Losing the hand drops the draw and forgets where neutral was.
  if (!hand) { if (gestureState!=="cooldown") gestureState="idle"; baseline=0; peakPull=0; wasClosed=false; return {aim:{x:.5,y:.5},pull:0,power:0,drawing:false}; }
  const sizeSpeed=lastHandTime?(hand.size-lastSize)/Math.max((now-lastHandTime)/1000,.001):0; lastSize=hand.size; lastHandTime=now;
  if (gestureState==="cooldown") { if (now<cooldownUntil) return {aim:hand.palm,pull:0,power:0,drawing:false}; gestureState="aiming"; }
  if (gestureState==="idle" || gestureState==="released") gestureState="aiming";
  if (gestureState==="aiming") {
    // Neutral hand distance re-centres slowly, so any seating position works and the
    // draw never depends on an absolute palm size.
    baseline = baseline ? baseline+(hand.size-baseline)*BASELINE_ALPHA : hand.size;
    if ((baseline-hand.size)/Math.max(baseline,.0001) >= DRAW_ENTER) { gestureState="drawing"; drawStarted=now; grabSize=baseline; peakPull=0; wasClosed=false; }
    return {aim:hand.palm,pull:0,power:0,drawing:gestureState==="drawing"};
  }
  // The entry threshold is subtracted so charging starts from zero, not a step.
  const depthChange=Math.max(0,(grabSize-hand.size)/Math.max(grabSize,.0001));
  const depth=clamp((depthChange-DRAW_ENTER)/Math.max(MAX_DEPTH_PULL-DRAW_ENTER,.0001));
  // Throwing forward shrinks the live draw, so power is charged from the peak reached.
  peakPull=Math.max(peakPull,depth);
  const thrust=clamp(sizeSpeed/MAX_RELEASE_SPEED);
  // Closedness sets nothing; draw distance and throw speed alone set power.
  const power=clamp(MIN_LAUNCH_POWER+(1-MIN_LAUNCH_POWER)*(PULL_WEIGHT*peakPull+(1-PULL_WEIGHT)*thrust));
  if (hand.closedness>=FIST_HELD_THRESHOLD) wasClosed=true;
  const openedFist=wasClosed&&hand.closedness<=OPEN_THRESHOLD;
  // Giving the draw back counts as a throw at any speed; a fast one just fires sooner.
  const released=peakPull>=MIN_CHARGED_PULL&&(peakPull-depth)>=RELEASE_FRACTION*peakPull;
  if ((thrust>=THRUST_TRIGGER||released||openedFist) && now-drawStarted>=MIN_DRAW_MS) { gestureState="released"; cooldownUntil=now+COOLDOWN_MS; return {aim:hand.palm,pull:depth,power,drawing:false,launch:true}; }
  // Only bail out when nothing meaningful was ever charged, so noise cannot misfire.
  if (depthChange<=DRAW_CANCEL&&peakPull<MIN_CHARGED_PULL) { gestureState="aiming"; peakPull=0; wasClosed=false; return {aim:hand.palm,pull:0,power:0,drawing:false}; }
  return {aim:hand.palm,pull:depth,power,drawing:true};
}
function burst(x,y) { for(let i=0;i<14;i++){const angle=Math.random()*Math.PI*2,speed=75+Math.random()*175;particles.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,life:.35+Math.random()*.35});} }
function launch(dir,power) { if(!flying && power>.04) { velocity={x:dir.x*Math.max(180,MAX_SPEED*power),y:dir.y*Math.max(180,MAX_SPEED*power)}; flying=true; shots++; } }
function update(dt) {
  particles=particles.filter(p=>{p.x+=p.vx*dt;p.y+=p.vy*dt;p.vy+=GRAVITY*.35*dt;p.life-=dt;return p.life>0;});
  updateLevels(dt);
  if(!flying)return; velocity.y+=GRAVITY*dt;projectile.x+=velocity.x*dt;projectile.y+=velocity.y*dt;
  if(projectile.y+18>=GROUND){projectile.y=GROUND-18;velocity.y*=-.38;velocity.x*=.72;if(Math.abs(velocity.y)<70)velocity={x:0,y:0};}
  targets.forEach(t=>{if(t.alive&&projectile.x+18>t.x&&projectile.x-18<t.x+t.w&&projectile.y+18>t.y&&projectile.y-18<t.y+t.h){t.alive=false;score+=t.value;velocity.x*=.65;velocity.y*=.65;burst(projectile.x,projectile.y);}});
  // Without this the shot never ends: `flying` stays set and every later launch is ignored.
  if(projectile.x-18>W||projectile.x+18<0) reloadSling();
  else if(velocity.x*velocity.x+velocity.y*velocity.y<1){ restTimer+=dt; if(restTimer>=RELOAD_DELAY) reloadSling(); }
  else restTimer=0;
}
function reloadSling(){ projectile={x:180,y:530}; velocity={x:0,y:0}; flying=false; restTimer=0; }
function text(value,x,y,size,color) { ctx.fillStyle=color||"#f9fdff";ctx.font="700 "+size+"px DM Sans, sans-serif";ctx.fillText(value,x,y); }
function panel(x,y,w,h,color) { ctx.fillStyle=color||"#15344a";ctx.beginPath();ctx.roundRect(x,y,w,h,14);ctx.fill();ctx.globalAlpha=.28;ctx.strokeStyle="#fff";ctx.stroke();ctx.globalAlpha=1; }
function draw(input) {
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#7ed3ff";ctx.fillRect(0,0,W,H);
  const cloud=(performance.now()/75)%(W+220);ctx.fillStyle="#ebf9ff";[[cloud-220,120,1],[cloud-650,205,.72],[cloud+360,85,.58]].forEach(c=>[[0,10,23],[28,0,31],[62,12,20]].forEach(p=>{ctx.beginPath();ctx.arc(c[0]+p[0]*c[2],c[1]+p[1]*c[2],p[2]*c[2],0,Math.PI*2);ctx.fill();}));
  ctx.fillStyle="#4fab57";ctx.fillRect(0,GROUND,W,H-GROUND);ctx.fillStyle="#388e46";ctx.fillRect(0,GROUND,W,7);
  const dir=direction(input.aim);if(!flying)projectile={x:180-dir.x*(input.drawing?MAX_PULL*input.pull:0),y:530-dir.y*(input.drawing?MAX_PULL*input.pull:0)};
  // Preview with the launch power, not the band stretch, or the arc lies.
  if(input.drawing&&!flying){ctx.fillStyle="#fff";let x=projectile.x,y=projectile.y,vx=dir.x*Math.max(180,MAX_SPEED*input.power),vy=dir.y*Math.max(180,MAX_SPEED*input.power);for(let i=0;i<28;i++){x+=vx*.08;y+=vy*.08;vy+=GRAVITY*.08;ctx.beginPath();ctx.arc(x,y,i<10?4:3,0,Math.PI*2);ctx.fill();}ctx.strokeStyle="#4c2b1c";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(168,512);ctx.lineTo(projectile.x,projectile.y);ctx.stroke();}
  ctx.strokeStyle="#64391f";ctx.lineWidth=18;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(160,GROUND);ctx.lineTo(178,530);ctx.lineTo(200,GROUND);ctx.stroke();
  if(input.drawing&&!flying){ctx.strokeStyle="#4c2b1c";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(192,512);ctx.lineTo(projectile.x,projectile.y);ctx.stroke();}
  targets.forEach(t=>{if(!t.alive)return;ctx.fillStyle="#2f683b";ctx.fillRect(t.x+5,t.y+6,t.w,t.h);panel(t.x,t.y,t.w,t.h,"#ffb740");text(String(t.value),t.x+11,t.y+t.h/2+7,20,"#202f3f");});
  particles.forEach(p=>{ctx.fillStyle="#ffb740";ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2,5*p.life),0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#94302e";ctx.beginPath();ctx.arc(projectile.x,projectile.y,21,0,Math.PI*2);ctx.fill();ctx.fillStyle="#ff5b46";ctx.beginPath();ctx.arc(projectile.x,projectile.y,18,0,Math.PI*2);ctx.fill();ctx.fillStyle="#ffe584";ctx.beginPath();ctx.arc(projectile.x+5,projectile.y-5,5,0,Math.PI*2);ctx.fill();
  panel(24,22,308,82);text("PULL POWER",43,52,16,"#b5d6e9");ctx.fillStyle="#34495b";ctx.fillRect(43,63,252,18);ctx.fillStyle="#ff5b46";ctx.fillRect(43,63,252*(input.power||0),18);text(String(Math.round((input.power||0)*100))+"%",290,78,20);
  panel(1007,22,249,82);text("SCORE",1027,53,16,"#b5d6e9");text(String(score).padStart(4,"0"),1026,88,42);text("SHOTS "+shots,1160,83,15,"#ffb740");
  panel(24,128,245,150,"#193f55");text("HOW TO PLAY",43,153,15,"#b5d6e9");[["1","Show your hand"],["2","Pull hand back"],["3","Throw forward"]].forEach((step,i)=>{const y=181+i*30;ctx.fillStyle=input.drawing&&i>0?"#ffb740":"#a9c9d9";ctx.beginPath();ctx.arc(53,y,10,0,Math.PI*2);ctx.fill();text(step[0],49,y+5,14,"#203040");text(step[1],72,y+5,16);});
  panel(24,604,330,34,"#193f55");text("GESTURE: "+(mouseMode?"MOUSE MODE":gestureState.toUpperCase()),40,627,16,input.drawing?"#ffb740":"#f9fdff");text("R restart   M mouse mode",25,694,15,"#18384d");
  panel(W/2-165,22,330,44,"#193f55");ctx.textAlign="center";text(`LEVEL ${level+1}/${LEVELS.length}  ·  ${LEVELS[level].name}`,W/2,50,20,"#ffb740");ctx.textAlign="left";
  if(completed) banner("ALL LEVELS CLEAR",`FINAL SCORE ${score} — PRESS R TO PLAY AGAIN`);
  else if(!targets.some(t=>t.alive)) banner("LEVEL CLEAR","NEXT LEVEL LOADING");
}
function banner(title,subtitle){
  panel(W/2-290,234,580,124);ctx.textAlign="center";
  text(title,W/2,300,44,"#ffb740");text(subtitle,W/2,338,16);ctx.textAlign="left";
}
function drawHand(landmarks){handCtx.clearRect(0,0,320,240);if(!landmarks)return;handCtx.strokeStyle="#79ffb0";handCtx.lineWidth=2;BONES.forEach(b=>{handCtx.beginPath();handCtx.moveTo(landmarks[b[0]].x*320,landmarks[b[0]].y*240);handCtx.lineTo(landmarks[b[1]].x*320,landmarks[b[1]].y*240);handCtx.stroke();});handCtx.fillStyle="#fff";landmarks.forEach(p=>{handCtx.beginPath();handCtx.arc(p.x*320,p.y*240,3,0,Math.PI*2);handCtx.fill();});}
async function startStream(deviceId){
  if(video.srcObject) video.srcObject.getTracks().forEach(track=>track.stop());
  // An exact deviceId pins the chosen camera; without one the browser picks its default.
  const constraints=deviceId?{deviceId:{exact:deviceId},width:{ideal:1280},height:{ideal:720}}
                            :{facingMode:"user",width:{ideal:1280},height:{ideal:720}};
  video.srcObject=await navigator.mediaDevices.getUserMedia({video:constraints,audio:false});
  await video.play();
  lastVideoTime=-1;  // Force the next frame through the landmarker after a switch.
}
async function populateCameras(){
  const cameras=(await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==="videoinput");
  const track=video.srcObject&&video.srcObject.getVideoTracks()[0];
  const activeId=track&&track.getSettings().deviceId;
  cameraSelect.replaceChildren(...cameras.map((camera,index)=>{
    const option=document.createElement("option");
    option.value=camera.deviceId; option.textContent=camera.label||("Camera "+(index+1));
    option.selected=camera.deviceId===activeId; return option;
  }));
  cameraPicker.hidden=cameras.length<2;  // Only worth showing when there is a choice.
}
async function enableCamera(){
  const button=document.querySelector("#start-camera"); button.disabled=true;
  try{
    // The camera is requested first so the permission prompt appears immediately,
    // and a slow or blocked CDN cannot stop the user from seeing their own video.
    cameraStatus.textContent="Requesting camera…"; trackingStatus.textContent="REQUESTING CAMERA";
    await startStream();
    // Device labels are only exposed once permission has been granted.
    await populateCameras();
    cameraStatus.textContent="Camera on. Loading hand tracking…"; trackingStatus.textContent="LOADING MODEL";
    const {FilesetResolver,HandLandmarker}=await import(TASKS_VISION+"/+esm");
    const vision=await FilesetResolver.forVisionTasks(TASKS_VISION+"/wasm");
    const options=delegate=>({baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",delegate},runningMode:"VIDEO",numHands:1,minHandDetectionConfidence:.55,minTrackingConfidence:.5});
    // Some drivers advertise WebGL but fail to build the GPU graph; CPU still runs fine.
    try{landmarker=await HandLandmarker.createFromOptions(vision,options("GPU"));}
    catch(gpuError){console.warn("GPU delegate failed, falling back to CPU.",gpuError);landmarker=await HandLandmarker.createFromOptions(vision,options("CPU"));}
    mouseMode=false;cameraStatus.textContent="Camera enabled. Show an open palm to begin.";trackingStatus.textContent="SHOW OPEN PALM";
  }catch(error){
    console.error(error);
    const denied=error&&(error.name==="NotAllowedError"||error.name==="NotFoundError");
    cameraStatus.textContent=denied
      ?"Camera blocked or not found. Allow camera access in the address bar, or use mouse mode."
      :"Hand tracking failed to load ("+(error&&error.message||error)+"). Mouse mode still works.";
    trackingStatus.textContent=denied?"CAMERA BLOCKED":"TRACKING UNAVAILABLE";
  }finally{button.disabled=false;}
}
function useMouse(){mouseMode=true;gestureState="aiming";cameraStatus.textContent="Mouse mode: hold on the game, drag, then release to launch.";trackingStatus.textContent="MOUSE MODE";}
function loop(now){
  const dt=Math.min((now-lastFrame)/1000,.05);lastFrame=now;let input;
  if(!mouseMode&&landmarker&&video.readyState>=2&&video.currentTime!==lastVideoTime){lastVideoTime=video.currentTime;const result=landmarker.detectForVideo(video,now),landmarks=result.landmarks&&result.landmarks[0];drawHand(landmarks);if(landmarks){input=control(analyze(landmarks),now);trackingStatus.textContent=gestureState.toUpperCase()+" · fist "+Math.round(smoothHand.closedness*100)+"%";}else{input=control(null,now);trackingStatus.textContent="SHOW OPEN PALM";}}
  if(mouseMode)input={aim:mouseAim,pull:mouseDown?mousePull:0,power:mouseDown?mousePull:0,drawing:mouseDown};if(input)activeInput=input;if(activeInput.launch){launch(direction(activeInput.aim),activeInput.power);activeInput.launch=false;}update(dt);draw(activeInput);requestAnimationFrame(loop);
}
canvas.addEventListener("pointerdown",event=>{if(mouseMode){mouseDown=true;canvas.setPointerCapture(event.pointerId);}});
canvas.addEventListener("pointermove",event=>{if(!mouseMode)return;const r=canvas.getBoundingClientRect();
  // Convert to canvas coordinates first: the canvas is CSS-scaled, so measuring the
  // drag in display pixels made every shot weaker on a smaller window.
  const cx=(event.clientX-r.left)/r.width*W, cy=(event.clientY-r.top)/r.height*H;
  mouseAim={x:clamp(cx/W),y:clamp(cy/H)};mousePull=clamp(Math.hypot(cx-180,cy-530)/350);});
canvas.addEventListener("pointerup",()=>{if(mouseMode&&mouseDown){mouseDown=false;launch(direction(mouseAim),Math.max(mousePull,.2));mousePull=0;}});
document.querySelector("#start-camera").addEventListener("click",enableCamera);document.querySelector("#mouse-mode").addEventListener("click",useMouse);
cameraSelect.addEventListener("change",async()=>{
  try{cameraStatus.textContent="Switching camera…";await startStream(cameraSelect.value);mouseMode=false;cameraStatus.textContent="Camera switched. Show an open palm to begin.";}
  catch(error){console.error(error);cameraStatus.textContent="Could not switch to that camera.";}
});window.addEventListener("keydown",event=>{if(event.key.toLowerCase()==="r")reset();if(event.key.toLowerCase()==="m")useMouse();});
reset();requestAnimationFrame(loop);
