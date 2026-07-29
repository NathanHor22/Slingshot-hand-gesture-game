import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

// Tune these values if a particular webcam or lighting setup needs adjustment.
const CLOSE_THRESHOLD = 0.68, OPEN_THRESHOLD = 0.38, CLOSE_STABLE_MS = 80, MIN_DRAW_MS = 150, COOLDOWN_MS = 500;
const MAX_DEPTH_PULL = 0.38, MAX_RELEASE_SPEED = 1.4, EMA_ALPHA = 0.30;
const W = 1280, H = 720, GROUND = 650, GRAVITY = 900, MAX_SPEED = 850, MAX_PULL = 110;
const PALM = [0, 5, 9, 13, 17], TIPS = [4, 8, 12, 16, 20];
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const canvas = document.querySelector("#game"), ctx = canvas.getContext("2d");
const video = document.querySelector("#camera"), overlay = document.querySelector("#hand-overlay"), handCtx = overlay.getContext("2d");
const cameraStatus = document.querySelector("#camera-status"), trackingStatus = document.querySelector("#tracking-status");
let landmarker, lastVideoTime = -1, gestureState = "idle", smoothHand, closeStarted = 0, drawStarted = 0, cooldownUntil = 0;
let grabSize = 0, heldPull = 0, lastSize = 0, lastHandTime = 0, mouseMode = false, mouseDown = false, mousePull = 0, mouseAim = {x:.5,y:.5};
let projectile, velocity, flying, score, shots, targets, particles, lastFrame = performance.now();
let activeInput = {aim:{x:.5,y:.5},pull:0,drawing:false};
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);

function reset() {
  projectile={x:180,y:530}; velocity={x:0,y:0}; flying=false; score=0; shots=0; particles=[];
  targets=[{x:850,y:570,w:46,h:80,value:100,alive:true},{x:970,y:530,w:54,h:120,value:150,alive:true},{x:1090,y:590,w:62,h:60,value:100,alive:true}];
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
  if (!hand) { if (gestureState!=="cooldown") gestureState="idle"; return {aim:{x:.5,y:.5},pull:0,drawing:false}; }
  const sizeSpeed=lastHandTime?(hand.size-lastSize)/Math.max((now-lastHandTime)/1000,.001):0; lastSize=hand.size; lastHandTime=now;
  if (gestureState==="cooldown") { if (now<cooldownUntil) return {aim:hand.palm,pull:0,drawing:false}; gestureState="aiming"; }
  if (gestureState==="idle" || gestureState==="released") gestureState="aiming";
  if (gestureState==="aiming") {
    if (hand.closedness>=CLOSE_THRESHOLD) { if (!closeStarted) closeStarted=now; if (now-closeStarted>=CLOSE_STABLE_MS) { gestureState="drawing"; drawStarted=now; grabSize=hand.size; heldPull=hand.closedness; } } else closeStarted=0;
    return {aim:hand.palm,pull:0,drawing:gestureState==="drawing"};
  }
  const depth=clamp((grabSize-hand.size)/Math.max(grabSize,.0001)/MAX_DEPTH_PULL);
  const pull=clamp(.6*hand.closedness+.4*depth);
  if (hand.closedness>OPEN_THRESHOLD) heldPull=hand.closedness;
  if (hand.closedness<=OPEN_THRESHOLD && now-drawStarted>=MIN_DRAW_MS) { gestureState="released"; cooldownUntil=now+COOLDOWN_MS; return {aim:hand.palm,pull,drawing:false,launch:true,power:clamp(.8*heldPull+.2*clamp(sizeSpeed/MAX_RELEASE_SPEED))}; }
  return {aim:hand.palm,pull,drawing:true};
}
function burst(x,y) { for(let i=0;i<14;i++){const angle=Math.random()*Math.PI*2,speed=75+Math.random()*175;particles.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,life:.35+Math.random()*.35});} }
function launch(dir,power) { if(!flying && power>.04) { velocity={x:dir.x*Math.max(180,MAX_SPEED*power),y:dir.y*Math.max(180,MAX_SPEED*power)}; flying=true; shots++; } }
function update(dt) {
  particles=particles.filter(p=>{p.x+=p.vx*dt;p.y+=p.vy*dt;p.vy+=GRAVITY*.35*dt;p.life-=dt;return p.life>0;});
  if(!flying)return; velocity.y+=GRAVITY*dt;projectile.x+=velocity.x*dt;projectile.y+=velocity.y*dt;
  if(projectile.y+18>=GROUND){projectile.y=GROUND-18;velocity.y*=-.38;velocity.x*=.72;if(Math.abs(velocity.y)<70)velocity={x:0,y:0};}
  targets.forEach(t=>{if(t.alive&&projectile.x+18>t.x&&projectile.x-18<t.x+t.w&&projectile.y+18>t.y&&projectile.y-18<t.y+t.h){t.alive=false;score+=t.value;velocity.x*=.65;velocity.y*=.65;burst(projectile.x,projectile.y);}});
}
function text(value,x,y,size,color) { ctx.fillStyle=color||"#f9fdff";ctx.font="700 "+size+"px DM Sans, sans-serif";ctx.fillText(value,x,y); }
function panel(x,y,w,h,color) { ctx.fillStyle=color||"#15344a";ctx.beginPath();ctx.roundRect(x,y,w,h,14);ctx.fill();ctx.globalAlpha=.28;ctx.strokeStyle="#fff";ctx.stroke();ctx.globalAlpha=1; }
function draw(input) {
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#7ed3ff";ctx.fillRect(0,0,W,H);
  const cloud=(performance.now()/75)%(W+220);ctx.fillStyle="#ebf9ff";[[cloud-220,120,1],[cloud-650,205,.72],[cloud+360,85,.58]].forEach(c=>[[0,10,23],[28,0,31],[62,12,20]].forEach(p=>{ctx.beginPath();ctx.arc(c[0]+p[0]*c[2],c[1]+p[1]*c[2],p[2]*c[2],0,Math.PI*2);ctx.fill();}));
  ctx.fillStyle="#4fab57";ctx.fillRect(0,GROUND,W,H-GROUND);ctx.fillStyle="#388e46";ctx.fillRect(0,GROUND,W,7);
  const dir=direction(input.aim);if(!flying)projectile={x:180-dir.x*(input.drawing?MAX_PULL*input.pull:0),y:530-dir.y*(input.drawing?MAX_PULL*input.pull:0)};
  if(input.drawing&&!flying){ctx.fillStyle="#fff";let x=projectile.x,y=projectile.y,vx=dir.x*Math.max(180,MAX_SPEED*input.pull),vy=dir.y*Math.max(180,MAX_SPEED*input.pull);for(let i=0;i<28;i++){x+=vx*.08;y+=vy*.08;vy+=GRAVITY*.08;ctx.beginPath();ctx.arc(x,y,i<10?4:3,0,Math.PI*2);ctx.fill();}ctx.strokeStyle="#4c2b1c";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(168,512);ctx.lineTo(projectile.x,projectile.y);ctx.stroke();}
  ctx.strokeStyle="#64391f";ctx.lineWidth=18;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(160,GROUND);ctx.lineTo(178,530);ctx.lineTo(200,GROUND);ctx.stroke();
  if(input.drawing&&!flying){ctx.strokeStyle="#4c2b1c";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(192,512);ctx.lineTo(projectile.x,projectile.y);ctx.stroke();}
  targets.forEach(t=>{if(!t.alive)return;ctx.fillStyle="#2f683b";ctx.fillRect(t.x+5,t.y+6,t.w,t.h);panel(t.x,t.y,t.w,t.h,"#ffb740");text(String(t.value),t.x+11,t.y+t.h/2+7,20,"#202f3f");});
  particles.forEach(p=>{ctx.fillStyle="#ffb740";ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2,5*p.life),0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#94302e";ctx.beginPath();ctx.arc(projectile.x,projectile.y,21,0,Math.PI*2);ctx.fill();ctx.fillStyle="#ff5b46";ctx.beginPath();ctx.arc(projectile.x,projectile.y,18,0,Math.PI*2);ctx.fill();ctx.fillStyle="#ffe584";ctx.beginPath();ctx.arc(projectile.x+5,projectile.y-5,5,0,Math.PI*2);ctx.fill();
  panel(24,22,308,82);text("PULL POWER",43,52,16,"#b5d6e9");ctx.fillStyle="#34495b";ctx.fillRect(43,63,252,18);ctx.fillStyle="#ff5b46";ctx.fillRect(43,63,252*input.pull,18);text(String(Math.round(input.pull*100))+"%",303,78,22);
  panel(1007,22,249,82);text("SCORE",1027,53,16,"#b5d6e9");text(String(score).padStart(4,"0"),1026,88,42);text("SHOTS "+shots,1160,83,15,"#ffb740");
  panel(24,128,245,180,"#193f55");text("HOW TO PLAY",43,153,15,"#b5d6e9");[["1","Show open palm"],["2","Close fist to grab"],["3","Move hand back"],["4","Open fist to launch"]].forEach((step,i)=>{const y=181+i*30;ctx.fillStyle=input.drawing&&i>0?"#ffb740":"#a9c9d9";ctx.beginPath();ctx.arc(53,y,10,0,Math.PI*2);ctx.fill();text(step[0],49,y+5,14,"#203040");text(step[1],72,y+5,16);});
  panel(24,604,330,34,"#193f55");text("GESTURE: "+(mouseMode?"MOUSE MODE":gestureState.toUpperCase()),40,627,16,input.drawing?"#ffb740":"#f9fdff");text("R restart   M mouse mode",25,694,15,"#18384d");
}
function drawHand(landmarks){handCtx.clearRect(0,0,320,240);if(!landmarks)return;handCtx.strokeStyle="#79ffb0";handCtx.lineWidth=2;BONES.forEach(b=>{handCtx.beginPath();handCtx.moveTo(landmarks[b[0]].x*320,landmarks[b[0]].y*240);handCtx.lineTo(landmarks[b[1]].x*320,landmarks[b[1]].y*240);handCtx.stroke();});handCtx.fillStyle="#fff";landmarks.forEach(p=>{handCtx.beginPath();handCtx.arc(p.x*320,p.y*240,3,0,Math.PI*2);handCtx.fill();});}
async function enableCamera(){
  try{cameraStatus.textContent="Loading hand tracking…";const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm");landmarker=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",delegate:"GPU"},runningMode:"VIDEO",numHands:1,minHandDetectionConfidence:.55,minTrackingConfidence:.5});video.srcObject=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});await video.play();mouseMode=false;cameraStatus.textContent="Camera enabled. Show an open palm to begin.";trackingStatus.textContent="SHOW OPEN PALM";}
  catch(error){cameraStatus.textContent="Camera could not start. Use mouse mode or allow camera access and try again.";trackingStatus.textContent="CAMERA BLOCKED";console.error(error);}
}
function useMouse(){mouseMode=true;gestureState="aiming";cameraStatus.textContent="Mouse mode: hold on the game, drag, then release to launch.";trackingStatus.textContent="MOUSE MODE";}
function loop(now){
  const dt=Math.min((now-lastFrame)/1000,.05);lastFrame=now;let input;
  if(!mouseMode&&landmarker&&video.readyState>=2&&video.currentTime!==lastVideoTime){lastVideoTime=video.currentTime;const result=landmarker.detectForVideo(video,now),landmarks=result.landmarks&&result.landmarks[0];drawHand(landmarks);if(landmarks){input=control(analyze(landmarks),now);trackingStatus.textContent=gestureState.toUpperCase()+" · fist "+Math.round(smoothHand.closedness*100)+"%";}else{input=control(null,now);trackingStatus.textContent="SHOW OPEN PALM";}}
  if(mouseMode)input={aim:mouseAim,pull:mouseDown?mousePull:0,drawing:mouseDown};if(input)activeInput=input;if(activeInput.launch){launch(direction(activeInput.aim),activeInput.power);activeInput.launch=false;}update(dt);draw(activeInput);requestAnimationFrame(loop);
}
canvas.addEventListener("pointerdown",event=>{if(mouseMode){mouseDown=true;canvas.setPointerCapture(event.pointerId);}});
canvas.addEventListener("pointermove",event=>{if(!mouseMode)return;const r=canvas.getBoundingClientRect();mouseAim={x:clamp((event.clientX-r.left)/r.width),y:clamp((event.clientY-r.top)/r.height)};mousePull=clamp(Math.hypot(event.clientX-r.left-r.width*.14,event.clientY-r.top-r.height*.74)/350);});
canvas.addEventListener("pointerup",()=>{if(mouseMode&&mouseDown){mouseDown=false;launch(direction(mouseAim),Math.max(mousePull,.2));mousePull=0;}});
document.querySelector("#start-camera").addEventListener("click",enableCamera);document.querySelector("#mouse-mode").addEventListener("click",useMouse);window.addEventListener("keydown",event=>{if(event.key.toLowerCase()==="r")reset();if(event.key.toLowerCase()==="m")useMouse();});
reset();requestAnimationFrame(loop);
