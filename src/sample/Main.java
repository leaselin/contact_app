package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.event.EventHandler;
import javafx.stage.WindowEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.opencv.core.Core;

public class Main extends Application {

    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    @Override
    public void start(Stage primaryStage) throws Exception{

        FXMLLoader loader = new FXMLLoader(getClass().getResource("mainWindow.fxml"));
        BorderPane root = (BorderPane)loader.load();

        root.setStyle("-fx-background-color: whitesmoke;");

        Scene scene = new Scene(root, 1200, 800);

        primaryStage.setTitle("Card Detection");
        primaryStage.setScene(scene);

        primaryStage.show();

        Controller controller = loader.getController();
        controller.init();

        primaryStage.setOnCloseRequest((we -> controller.setClosed()));
    }


    public static void main(String[] args) {
        launch(args);
    }
}
