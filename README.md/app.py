from flask import Flask, request, render_template, send_from_directory, jsonify
from pathlib import Path
import cv2, numpy as np, time, json

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / 'uploads'; OUTPUTS = BASE / 'outputs'; MODELS = BASE / 'models'
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
app = Flask(__name__)
ALLOWED = {'.jpg','.jpeg','.png','.mp4','.avi','.mov','.mkv'}

# COCO classes relevant to road safety. COCO includes stop sign and traffic light.
ROAD_CLASSES = {'person','bicycle','car','motorcycle','bus','truck','stop sign','traffic light'}

def load_calibration():
    p = MODELS / 'calibration_params.npz'
    if p.exists():
        d = np.load(p)
        return d['K'], d['D']
    return None, None

def undistort(frame):
    K, D = load_calibration()
    if K is None: return frame
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    return cv2.undistort(frame, K, D, None, newK)

def preprocess(frame):
    frame = cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

def lane_overlay(frame):
    out = frame.copy(); h,w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    mask = np.zeros_like(edges)
    poly = np.array([[(int(.05*w),h),(int(.42*w),int(.55*h)),(int(.58*w),int(.55*h)),(int(.98*w),h)]])
    cv2.fillPoly(mask,poly,255)
    lines = cv2.HoughLinesP(cv2.bitwise_and(edges,mask),1,np.pi/180,40,minLineLength=45,maxLineGap=80)
    left=[]; right=[]
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2=map(int,l); dx=x2-x1
            if dx==0: continue
            slope=(y2-y1)/dx
            if -3 < slope < -0.25: left.append(l)
            elif 0.25 < slope < 3: right.append(l)
    for group, color in [(left,(255,180,0)),(right,(0,255,0))]:
        for x1,y1,x2,y2 in group[:10]: cv2.line(out,(x1,y1),(x2,y2),color,5)
    cv2.polylines(out,poly,True,(255,200,0),2)
    return out

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)

def safety_overlay(frame, detections):
    h,w=frame.shape[:2]
    pedestrians=[]
    for d in detections:
        if d['class']=='person':
            cx=(d['box'][0]+d['box'][2])/2; by=d['box'][3]
            if by > .55*h and .10*w < cx < .90*w: pedestrians.append(d)
    status='SAFE'; msg='Road scene monitored'
    if pedestrians:
        status='HIGH RISK'; msg='PEDESTRIAN IN ROAD CORRIDOR'
        cv2.rectangle(frame,(20,20),(510,72),(0,0,220),-1)
        cv2.putText(frame,msg,(32,55),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
    else:
        cv2.rectangle(frame,(20,20),(340,68),(30,130,30),-1)
        cv2.putText(frame,status,(32,52),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
    return status,msg

def detect_image(path, outpath):
    frame=cv2.imread(str(path));
    if frame is None: raise ValueError('Unable to read image')
    t0=time.perf_counter(); frame=preprocess(undistort(frame)); lane=lane_overlay(frame)
    detections=[]
    if YOLO is not None:
        model_path = MODELS/'yolov8n.pt'
        model = YOLO(str(model_path) if model_path.exists() else 'yolov8n.pt')
        r=model.predict(lane, conf=.35, verbose=False)[0]
        names=r.names
        if r.boxes is not None:
            for box,conf,cls in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
                name=names[int(cls)]
                if name in ROAD_CLASSES:
                    b=list(map(int,box)); detections.append({'class':name,'conf':float(conf),'box':b})
                    cv2.rectangle(lane,(b[0],b[1]),(b[2],b[3]),(0,255,0),2)
                    cv2.putText(lane,f'{name} {conf:.2f}',(b[0],max(22,b[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
    status,msg=safety_overlay(lane,detections)
    ms=(time.perf_counter()-t0)*1000
    cv2.putText(lane,f'Latency: {ms:.1f} ms | Objects: {len(detections)}',(20,lane.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.imwrite(str(outpath),lane)
    return {'mode':'image','objects':len(detections),'latency_ms':ms,'status':status,'message':msg,'output':outpath.name}

def track_video(path,outpath):
    if YOLO is None: raise RuntimeError('Install ultralytics first')
    model=YOLO(str(MODELS/'yolov8n.pt') if (MODELS/'yolov8n.pt').exists() else 'yolov8n.pt')
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w>1280: w=1280; h=int(h*1280/(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280))
    writer=cv2.VideoWriter(str(outpath),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    count=0; total_ms=0; max_tracks=0; hazards=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        t=time.perf_counter(); frame=preprocess(undistort(frame)); frame=lane_overlay(frame)
        r=model.track(frame,persist=True,tracker='bytetrack.yaml',conf=.35,verbose=False)[0]
        n=0; persons=[]
        if r.boxes is not None:
            for i,(box,conf,cls) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy())):
                name=r.names[int(cls)]
                if name not in ROAD_CLASSES: continue
                b=list(map(int,box)); tid=int(r.boxes.id[i].item()) if r.boxes.id is not None else i+1; n+=1
                cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),(0,255,0),2)
                cv2.putText(frame,f'ID {tid} | {name} {conf:.2f}',(b[0],max(20,b[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
                if name=='person' and b[3]>.55*h: persons.append(b)
        if persons:
            hazards+=1; cv2.rectangle(frame,(20,20),(520,72),(0,0,220),-1); cv2.putText(frame,'HAZARD: PEDESTRIAN NEAR ROAD',(30,55),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        total_ms+=(time.perf_counter()-t)*1000; count+=1; max_tracks=max(max_tracks,n)
        avg_fps=1000/((total_ms/count)+1e-9)
        cv2.putText(frame,f'Frame: {count} | FPS: {avg_fps:.1f} | Tracks: {n}',(20,h-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        writer.write(cv2.resize(frame,(w,h)))
    cap.release(); writer.release()
    return {'mode':'video','frames':count,'avg_latency_ms':total_ms/max(count,1),'avg_fps':1000/(total_ms/max(count,1)+1e-9),'max_tracks':max_tracks,'hazard_frames':hazards,'output':outpath.name}

@app.route('/')
def index(): return render_template('index.html')
@app.route('/process',methods=['POST'])
def process():
    f=request.files.get('file')
    if not f or not f.filename: return jsonify({'error':'Choose an image or video'}),400
    ext=Path(f.filename).suffix.lower()
    if ext not in ALLOWED: return jsonify({'error':'Unsupported file type'}),400
    src=UPLOADS/f.filename; f.save(src)
    try:
        if ext in {'.jpg','.jpeg','.png'}:
            out=OUTPUTS/(src.stem+'_annotated.jpg'); result=detect_image(src,out)
        else:
            out=OUTPUTS/(src.stem+'_tracked.mp4'); result=track_video(src,out)
        (OUTPUTS/'metrics.json').write_text(json.dumps(result,indent=2))
        result['url']='/outputs/'+out.name
        return jsonify(result)
    except Exception as e: return jsonify({'error':str(e)}),500
@app.route('/outputs/<path:name>')
def outputs(name): return send_from_directory(OUTPUTS,name)

if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=True)
