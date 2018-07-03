package sample;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.scene.control.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;


public class Controller {

    // FXML buttons
    @FXML
    private Button cameraButton;
    @FXML
    private TextArea numberOfCards;
    @FXML
    private Slider blur;
    @FXML
    private Slider threshold;
    // the FXML area for showing the current frame
    @FXML
    private ImageView originalFrame;
    @FXML
    private ImageView graysFrame;
    @FXML
    private ImageView threshFrame;
    // checkboxes for enabling/disabling a classifier
    @FXML
    private CheckBox haarClassifier;
    @FXML
    private CheckBox lbpClassifier;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private VideoCapture capture;
    // a flag to change the button behavior
    private boolean cameraActive;

    // face cascade classifier
    private CascadeClassifier faceCascade;
    private int absoluteFaceSize;


    protected void init()
    {
        this.capture = new VideoCapture();
        this.faceCascade = new CascadeClassifier();
        this.absoluteFaceSize = 0;

        // set a fixed width for the frame
        originalFrame.setFitWidth(800);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);
    }

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera()
    {


        if (!this.cameraActive)
        {
            // disable setting checkboxes
            this.haarClassifier.setDisable(true);
            this.lbpClassifier.setDisable(true);

            // start the video capture
            //this.capture.open(0);
            this.capture.open("resources/Cards Black 1.mp4");

            // is the video stream available?
            if (this.capture.isOpened())
            {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run()
                    {
                        // effectively grab and process a single frame
                        Mat frame = grabFrame();
                        Mat frame2 = grabThreshFrame();
                        Mat grayFrame = grabGrayFrame();
                        // convert and show the frame
                        Image imageToShow = Utils.mat2Image(frame);
                        Image imageToShow2 = Utils.mat2Image(frame2);
                        Image imageToShow3 = Utils.mat2Image(grayFrame);


                        if (imageToShow != null) {
                            updateImageView(originalFrame, imageToShow);
                        }
                        if (imageToShow3 != null) {
                            updateImageView(graysFrame, imageToShow3);
                        }
                        if (imageToShow2 != null) {
                            updateImageView(threshFrame, imageToShow2);
                        }
                        frame2.release();
                        frame.release();
                        grayFrame.release();
                    }

                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            }
            else
            {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        }
        else
        {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable classifiers checkboxes
            this.haarClassifier.setDisable(false);
            this.lbpClassifier.setDisable(false);

            // stop the timer
            this.stopAcquisition();
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Mat grabFrame()
    {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty())
                {
                    // card detection
                    this.detectAndDisplay(frame);
                    //Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
                }

            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return frame;
    }

    private Mat grabGrayFrame()
    {
        Mat grayFrame = new Mat();
        Mat blurMat = new Mat();


        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(grayFrame);

                // if the frame is not empty, process it
                if (!grayFrame.empty())
                {
                    Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.blur(grayFrame, blurMat, new Size(blur.getValue(), blur.getValue()));
                }
            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return blurMat;
    }

    private Mat grabThreshFrame()
    {
        Mat frame2 = new Mat();
        Mat threshMat = new Mat();

        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(frame2);

                // if the frame is not empty, process it
                if (!frame2.empty())
                {
                    Imgproc.cvtColor(frame2, frame2, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.blur(frame2, frame2, new Size(3, 3));
                    //Imgproc.threshold(frame2, threshMat, threshold.getValue(), 255, Imgproc.THRESH_BINARY);
                    Imgproc.threshold(frame2, threshMat, 180, 255, Imgproc.THRESH_BINARY);
                }
            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return threshMat;
    }


    /**
     * Method for face detection and tracking
     *
     * @param frame
     *            it looks for faces in this frame
     */
    private void detectAndDisplay(Mat frame)
    {
        Mat edges = new Mat();
        Mat blurMat = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        int counter = 0;
        //MatOfRect cards = new MatOfRect();

        Imgproc.blur(frame, blurMat, new Size(3, 3));

        // canny detector, with ratio of lower:upper threshold of 3:1
        Imgproc.Canny(blurMat, edges, this.threshold.getValue(), this.threshold.getValue() * 3, 3, true);

        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = getMaxArea(contours);

        for (int i = 0; i != contours.size(); ++i) {
            double area = Imgproc.contourArea(contours.get(i));
            if(area == maxArea) {

                // Generate the convex hull of this contour
                MatOfInt hullInt = new MatOfInt();
                Imgproc.convexHull(contours.get(i), hullInt);
                MatOfPoint hullPoint = OpenCVUtil.getNewContourFromIndices(contours.get(i), hullInt);

                // Use approxPolyDP to simplify the convex hull
                MatOfPoint2f polygon = new MatOfPoint2f();
                Imgproc.approxPolyDP(OpenCVUtil.convert(hullPoint), polygon, 20, true);
                List<MatOfPoint> tmp = new ArrayList<>();
                tmp.add(OpenCVUtil.convert(polygon));
                Imgproc.drawContours(frame, tmp, 0, new Scalar(0, 255, 0), 3);

                counter++;
                String cardsDetected = Integer.toString(counter);
                numberOfCards.setText(cardsDetected);
            }
        }



    }

    private double getMaxArea(List<MatOfPoint> contours) {
        double maxArea=0;

        for (int idx = 0; idx != contours.size(); ++idx)
        {
            Mat contour = contours.get(idx);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea)
            {
                maxArea = contourArea;
            }

        }

        return maxArea;
    }

    /*
    private MatOfPoint getMaxContour(List<MatOfPoint> contours) {
        double maxArea=0;
        int maxAreaIdx=0;

        for (int idx = 0; idx != contours.size(); ++idx)
        {
            Mat contour = contours.get(idx);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea)
            {
                maxArea = contourArea;
                maxAreaIdx = idx;
            }

        }

        return contours.get(maxAreaIdx);
    }
    */



    /**
     * The action triggered by selecting the Haar Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void haarSelected(Event event)
    {
        // check whether the lpb checkbox is selected and deselect it
        if (this.lbpClassifier.isSelected())
            this.lbpClassifier.setSelected(false);

        this.checkboxSelection("resources/haarcascades/haarcascade_eye.xml");
    }

    /**
     * The action triggered by selecting the LBP Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void lbpSelected(Event event)
    {
        // check whether the haar checkbox is selected and deselect it
        if (this.haarClassifier.isSelected())
            this.haarClassifier.setSelected(false);

        this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
    }

    /**
     * Method for loading a classifier trained set from disk
     *
     * @param classifierPath
     *            the path on disk where a classifier trained set is located
     */
    private void checkboxSelection(String classifierPath)
    {
        // load the classifier(s)
        this.faceCascade.load(classifierPath);

        // now the video capture can start
        this.cameraButton.setDisable(false);
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    private void stopAcquisition()
    {
        if (this.timer!=null && !this.timer.isShutdown())
        {
            try
            {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            }
            catch (InterruptedException e)
            {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened())
        {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view
     *            the {@link ImageView} to update
     * @param image
     *            the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image)
    {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed()
    {
        this.stopAcquisition();
    }

    public void setCameraButton(Button cameraButton) {
        this.cameraButton = cameraButton;
    }

    public void setOriginalFrame(ImageView originalFrame) {
        this.originalFrame = originalFrame;
    }

    public void setThreshFrame(ImageView threshFrame) {
        this.threshFrame = threshFrame;
    }

    public void setHaarClassifier(CheckBox haarClassifier) {
        this.haarClassifier = haarClassifier;
    }

    public void setLbpClassifier(CheckBox lbpClassifier) {
        this.lbpClassifier = lbpClassifier;
    }
}
