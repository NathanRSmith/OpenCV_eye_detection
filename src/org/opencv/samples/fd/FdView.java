package org.opencv.samples.fd;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.RectF;
import android.util.Log;
import android.view.Display;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.WindowManager;

class FdView extends SampleCvViewBase {
    private static final String   TAG = "Sample::FdView";
    private Mat                   mRgba;
    private Mat                   mGray;
    private File                  mCascadeFile;
    private CascadeClassifier     mJavaDetector;
    private DetectionBasedTracker mNativeDetector;

    private static final Scalar   FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    
    public static final int       JAVA_DETECTOR     = 0;
    public static final int       NATIVE_DETECTOR   = 1;
    
    private int                   mDetectorType     = JAVA_DETECTOR;

    private float                 mRelativeFaceSize = 0;
    private int					  mAbsoluteFaceSize = 0;
    
    protected boolean			  calibrationMode;
    private int				  	  calibrationPhase = 0;
    private Paint				  calibrationPointPaintOuter;
    private Paint				  calibrationPointPaintInner;
    private ArrayList<Point>      calibrationPoints = new ArrayList<Point>();
    private Paint				  calibrationModeTextPaint;
    
    private RectF				  brightnessRect;
    private Paint				  brightnessPaint;
    
    private int					  width;
    private int					  height;
    
    private Point				  gazeLocation = new Point(0,0);
    private float				  gazeUncertainty;
    
    private HashMap<String,Object> tabletDims;
	private double faceDist;
	private double eyeDist;
	private double pitchmin;
	private double pitchmax;
	private double lefteye_leftmax;
	private double lefteye_rightmax;
	private double righteye_leftmax;
	private double righteye_rightmax;
	private Paint gazePaint;
    
    public void setMinFaceSize(float faceSize)
    {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
    }
    
    public void setDetectorType(int type)
    {
    	if (mDetectorType != type)
    	{
    		mDetectorType = type;
    		
    		if (type == NATIVE_DETECTOR)
    		{
    			Log.i(TAG, "Detection Based Tracker enabled");
    			mNativeDetector.start();
    		}
    		else
    		{
    			Log.i(TAG, "Cascade detector enabled");
    			mNativeDetector.stop();
    		}
    	}
    }

    public FdView(Context context) {
        super(context);

        try {
        	
        	tabletDims = new HashMap<String, Object>();
        	tabletDims.put("units", "cm");
        	tabletDims.put("width", 25.0);
        	tabletDims.put("height", 17.0);
        	tabletDims.put("border_top", 1.5);
        	tabletDims.put("border_bottom", 1.5);
        	tabletDims.put("border_left", 1.5);
        	tabletDims.put("border_right", 1.5);
        	tabletDims.put("resolution_width", 1280);
        	tabletDims.put("resolution_height", 800);
        	tabletDims.put("camera_postition_x", 12.5);
        	tabletDims.put("camera_postition_y", .65);
        	
        	faceDist = 30.0;
        	eyeDist = 6.0;
        	
        	// angles
        	pitchmin = Math.atan2((Double)tabletDims.get("border_top"), faceDist);
        	pitchmax = Math.atan2((Double)tabletDims.get("height") - (Double)tabletDims.get("border_top"), faceDist);
        	lefteye_leftmax = Math.atan2( ( (Double)tabletDims.get("width") - eyeDist -
        									((Double)tabletDims.get("border_left") + (Double)tabletDims.get("border_right"))) / 2, faceDist);
        	lefteye_rightmax = Math.atan2( ( (Double)tabletDims.get("width") + eyeDist -
											((Double)tabletDims.get("border_left") + (Double)tabletDims.get("border_right"))) / 2, faceDist);
        	righteye_leftmax = lefteye_rightmax;
        	righteye_rightmax = lefteye_leftmax;
        	
        	
        	calibrationMode = true;
        	
        	calibrationPointPaintOuter = new Paint();
        	calibrationPointPaintOuter.setStyle(Paint.Style.STROKE);
        	calibrationPointPaintOuter.setStrokeWidth(5);
        	calibrationPointPaintOuter.setColor(Color.RED);
        	
        	calibrationPointPaintInner = new Paint();
        	calibrationPointPaintInner.setStyle(Paint.Style.FILL);
        	calibrationPointPaintInner.setColor(Color.RED);
        	
        	calibrationModeTextPaint = new Paint();
        	calibrationModeTextPaint.setColor(Color.BLUE);
        	calibrationModeTextPaint.setTextSize(50);
        	
        	Display disp = ((WindowManager) context.getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay();
        	int w = disp.getWidth();
        	int h = disp.getHeight();
        	width = w;
        	height = h;
        	
        	Log.i("display dimensions", "display dimensions: "+w+", "+h);
        	
        	calibrationPoints.add(new Point(w/2,0));
        	calibrationPoints.add(new Point(0,h/2));
        	calibrationPoints.add(new Point(w,h/2));
        	calibrationPoints.add(new Point(w/2,h));
        	calibrationPoints.add(new Point(w/2,h/2));
        	
        	brightnessRect = new RectF(0,0,w,h);
        	brightnessPaint = new Paint();
        	brightnessPaint.setStyle(Paint.Style.FILL);
        	brightnessPaint.setColor(Color.WHITE);
        	brightnessPaint.setAlpha(150);
        	
        	gazePaint = new Paint();
        	gazePaint.setStyle(Paint.Style.FILL);
        	gazePaint.setColor(Color.CYAN);
        	
        	
//        	InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
        	InputStream is = context.getResources().openRawResource(R.raw.haarcascade_mcs_righteye);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
//            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            mCascadeFile = new File(cascadeDir, "haarcascade_mcs_righteye.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (mJavaDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
            
            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }
    
    public void drawGazeLocation(Canvas canvas, Point loc, float uncertainty) {
    	canvas.drawCircle(loc.x, loc.y, uncertainty, gazePaint);
    }
    
    public Point findGazeLocation(Double lefteye_yaw, Double lefteye_pitch, Double righteye_yaw, Double righteye_pitch, String units) {
    	if( units == "deg" ) {
    		lefteye_yaw = lefteye_yaw * Math.PI / 180.0;
    		lefteye_pitch = lefteye_pitch * Math.PI / 180.0;
    		righteye_yaw = righteye_yaw * Math.PI / 180.0;
    		righteye_pitch = righteye_pitch * Math.PI / 180.0;
    	}
    	
    	Double lefteye_x = (Double)tabletDims.get("width")/2 - eyeDist/2 + faceDist*Math.tan(lefteye_yaw);
    	Double righteye_x = (Double)tabletDims.get("width")/2 - eyeDist/2 + faceDist*Math.tan(righteye_yaw);
    	Double ave_x = (lefteye_x + righteye_x) / 2;
    	
    	// if the gaze location is outside of screen area
    	if( ave_x < (Double)tabletDims.get("border_left") || ave_x > (Double)tabletDims.get("width") - (Double)tabletDims.get("border_right") ) {
    		return new Point(-1,-1);
    	}
    	
    	Double lefteye_y = faceDist * Math.tan(lefteye_pitch);
        Double righteye_y = faceDist * Math.tan(righteye_pitch);
        Double ave_y = (lefteye_y + righteye_y) / 2;
        
        // if the gaze location is outside of screen area
        if( ave_y < (Double)tabletDims.get("border_top") || ave_y > (Double)tabletDims.get("height") - (Double)tabletDims.get("border_bottom") ) {
        	return new Point(-1,-1);
        }
        
        Double px_x = ( (ave_x - (Double)tabletDims.get("border_left")) / ((Double)tabletDims.get("width") - (Double)tabletDims.get("border_left") - (Double)tabletDims.get("border_right")) ) * (Double)tabletDims.get("resolution_width");
        Double px_y = ( (ave_y - (Double)tabletDims.get("border_top")) / ((Double)tabletDims.get("height") - (Double)tabletDims.get("border_top") - (Double)tabletDims.get("border_bottom")) ) * (Double)tabletDims.get("resolution_height");
    	
    	return new Point(px_x.intValue(), px_y.intValue());
    }
    
    public Point findGazeLocation(Double lefteye_yaw, Double lefteye_pitch, Double righteye_yaw, Double righteye_pitch) {
    	return findGazeLocation(lefteye_yaw, lefteye_pitch, righteye_yaw, righteye_pitch, "rad");
    }
    

    @Override
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
        }

        super.surfaceCreated(holder);
	}

	@Override
    protected Bitmap processFrame(VideoCapture capture) {
        capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);

        if (mAbsoluteFaceSize == 0)
        {
        	int height = mGray.rows();
        	if (Math.round(height * mRelativeFaceSize) > 0);
        	{
        		mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
        	}
        	mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }
        
        MatOfRect faces = new MatOfRect();
        
        if (mDetectorType == JAVA_DETECTOR)
        {
        	if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2 // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        , new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR)
        {
        	if (mNativeDetector != null)
        		mNativeDetector.detect(mGray, faces);
        }
        else
        {
        	Log.e(TAG, "Detection method is not selected!");
        }
        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mRgba, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        return bmp;
    }
	
	@Override
	protected void drawCalibrationPoint(Canvas canvas) {
		if( calibrationMode == true) {
			Point p = calibrationPoints.get(calibrationPhase);
			canvas.drawCircle(p.x, p.y, 15, calibrationPointPaintOuter);
			canvas.drawCircle(p.x, p.y, 5, calibrationPointPaintInner);
			
			canvas.drawText("Calibration Mode", 500, 500, calibrationModeTextPaint);
		}
	}
	
	@Override
	public boolean onTouchEvent(MotionEvent e) {
		
		switch (e.getAction()) {
			case MotionEvent.ACTION_DOWN:
		
				if( calibrationMode == true ) {
					startNextCalibrationPhase();
				}
				
		}
		
		return true;
	}
	
	/* Moves to the next calibration phase.  If the current phase is the final one, 
	 * calibration mode is complete and reset to false.
	 */
	private void startNextCalibrationPhase() {
		if( calibrationMode == true ) {
			calibrationPhase += 1;
			if( calibrationPhase >= calibrationPoints.size() ) {
				calibrationMode = false;
				calibrationPhase = 0;
			}
		}
	}
	
	@Override
	protected void drawBrightnessOverlay(Canvas canvas) {
		canvas.drawRect(brightnessRect, brightnessPaint);
	}

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mCascadeFile != null)
            	mCascadeFile.delete();
            if (mNativeDetector != null)
            	mNativeDetector.release();

            mRgba = null;
            mGray = null;
            mCascadeFile = null;
        }
    }
}
