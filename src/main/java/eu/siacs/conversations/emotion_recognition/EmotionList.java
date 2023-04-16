package eu.siacs.conversations.emotion_recognition;

import java.util.ArrayList;

public class EmotionList<T> extends ArrayList<float[]> {
    private final int limit;
    private int emotions;
    public EmotionList(int limit, int emotions) {
        this.limit = limit;
        this.emotions = emotions;
    }

    public void setEmotions(int emotions) {
        this.emotions = emotions;
        super.clear();
    }

    public boolean add(float[] array) {
        while (this.size() >= limit) {
            this.remove(0);
        }
        return super.add(array);
    }

    public float[] getEmotionAverages() {
        if (this.size() == 0) {
            return null;
        }
        float[] averages = new float[emotions];
        for (int i = 0; i < emotions; i++) {
            averages[i] = 0;
        }

        for (float[] detection : this) {
            for (int i = 0; i < emotions; i++) {
                if (i < detection.length) {
                    averages[i] += detection[i];
                }
            }
        }

        for (int i = 0; i < emotions; i++) {
            averages[i] /= this.size();
        }

        return averages;
    }

    public void removeLast() {
        if (this.size() != 0) {
            this.remove(0);
        }
    }

    public float[] getTail() {
        if (this.size() != 0) {
            return this.get(this.size() - 1);
        }
        else {
            return new float[] {};
        }
    }

    public int getIndexOfMax() {
        float[] averages = getEmotionAverages();
        if (averages == null || averages.length == 0) {
            return -1;
        }
        int index = 0;
        float max = averages[0];

        for (int i = 1; i < averages.length; i++) {
            index = max > averages[i] ? index : i;
            max = Math.max(max, averages[i]);
        }
        return index;
    }
}