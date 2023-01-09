package task2;

public class TimeResult<T> {
    double time;
    T result;

    TimeResult(double time, T result) {
        this.time = time;
        this.result = result;
    }
}