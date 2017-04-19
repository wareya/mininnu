#include <stddef.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "genann.h"

// Released to the public domain under US law and publicly licensed under the
// ISC license. No warranty provided, express or implied; the authors will not
// be liable for any damages arising from the use of this software.

struct image {
    int w, h, n;
    unsigned char * data;
    inline double pixel(int x, int y, int c)
    {
        // For ~problems~, we reflect instead of clamp. This is definitely not a good thing, and a real world implementation would solve the ~problems~ some other way.
        x = (x<0)?-x:((x>=w)?2*w-x-2:x);
        y = (y<0)?-y:((y>=h)?2*h-y-2:y);
        c = (c<0)? 0:((c>=n)?n-1:c);
        
        return data[(c)+(x*n)+(y*w*n)]/255.0;
    }
};

unsigned char tochar(double n)
{
    n = round(n*255.0);
    if(n<0) n = 0;
    if(n>=255.0) n = 255.0;
    return n;
}

// Takes a 32 neuron input field and balances it rotationally and laterally
void balance(double* field)
{
    field[ 0] = field[0]/8 + field[ 9]/8 + field[31]/8 + field[ 4]/8 + field[3]/8 + field[27]/8 + field[28]/8 + field[22]/8;
    field[ 1] = field[1]/8 + field[15]/8 + field[30]/8 + field[16]/8 + field[2]/8 + field[21]/8 + field[29]/8 + field[10]/8;
    field[ 2] = field[1];
    field[ 3] = field[0];
    
    field[ 4] = field[0];
    field[ 5] = field[5]/4 + field[8]/4 + field[23]/4 + field[26]/4;
    field[ 6] = field[6]/8 + field[7]/8 + field[11]/8 + field[14]/8 + field[17]/8 + field[20]/8 + field[24]/8 + field[25]/8;
    field[ 7] = field[6];
    field[ 8] = field[5];
    field[ 9] = field[0];
    
    field[10] = field[1];
    field[11] = field[6];
    field[12] = field[12]/4 + field[13]/4 + field[18]/4 + field[19]/4;
    field[13] = field[12];
    field[14] = field[6];
    field[15] = field[1];
    
    field[16] = field[1];
    field[17] = field[6];
    field[18] = field[12];
    field[19] = field[12];
    field[20] = field[6];
    field[21] = field[1];
    
    field[22] = field[0];
    field[23] = field[5];
    field[24] = field[6];
    field[25] = field[6];
    field[26] = field[5];
    field[27] = field[0];
    
    field[28] = field[0];
    field[29] = field[1];
    field[30] = field[1];
    field[31] = field[0];
}

int main()
{
    // Should be preblurred for performance and flexibility reasons
    image myimage;
    myimage.data = stbi_load("intest.png", &(myimage.w), &(myimage.h), &(myimage.n), 0);
    if(!myimage.data) return printf("stbi_load failed: %s\n", stbi_failure_reason()), 0;
    
    // Should not have been downscaeld in a way that destroys high frequency phase information (i.e. don't use a box filter)
    image myimage2;
    myimage2.data = stbi_load("toupscale.png", &(myimage2.w), &(myimage2.h), &(myimage2.n), 0);
    if(!myimage2.data) return printf("stbi_load failed: %s\n", stbi_failure_reason()), 0;
    
    // 32 inputs, 4 hidden layers of 4 nodes each
    genann *ann = genann_init(32, 4, 8, 1);
    ann->activation_hidden = genann_act_tanh;
    
    // We force the input weights to be symmetrical for each neuron in the first hidden layer 
    // This step isn't necessary, but it prevents the network from making "unbalanced" outputs out of balanced inputs
    // We also do this during training.
    // The specific way we balance the field isn't ideal at all, it prevents the field from having z-axis symmetry as well, which basically means it can't be laterally mirrored with one half negated.
    for(int i = 0; i < 8; i++)
        balance(&ann->weight[(32+1)*i+1]); // each hidden layer has 32+1 inputs, the first is the hidden constant input
    // It should be noted that this balancing step also reduces the number of unique weights in the network from 489 (?) to 265 (?).
    
    int w, h, n;
    
    w = myimage.w;
    h = myimage.h;
    n = myimage.n;
    
    // test speedup stuff
    const int skip = 1;
    const int earliness = 1;
    const int channels = n;
    
    bool useavg = true; // normalize inputs to have an average of 1.0 before feeding to the ANN
    bool hardavg = true; // normalize inputs to be centered on 0 before feeding to the ANN
    
    // learning rate; network will actually fail to train properly if this is set to a bad value
    double rate;
    if(!hardavg || !useavg)
        rate = 0.2;
    else
        rate = 0.5;
    
    // how many times to train the net on the image
    const int iterations = 32;
    
    for(int iff = 0; iff < iterations; iff++)
    {
        for (auto y = 0; y < h/earliness; y+=skip)
        {
            for (auto x = 0; x < w/earliness; x+=skip)
            {
                for (auto c = 0; c < channels; c++)
                {
                    // our input is a 6x6 grid with the corners cut off
                    double avg = 0.0;
                    double output[1];
                    double input[32];
                    input[ 0] = myimage.pixel(x-3, y-5, c);
                    input[ 1] = myimage.pixel(x-1, y-5, c);
                    input[ 2] = myimage.pixel(x+1, y-5, c);
                    input[ 3] = myimage.pixel(x+3, y-5, c);
                    
                    input[ 4] = myimage.pixel(x-5, y-3, c);
                    input[ 5] = myimage.pixel(x-3, y-3, c);
                    input[ 6] = myimage.pixel(x-1, y-3, c);
                    input[ 7] = myimage.pixel(x+1, y-3, c);
                    input[ 8] = myimage.pixel(x+3, y-3, c);
                    input[ 9] = myimage.pixel(x+5, y-3, c);
                    
                    input[10] = myimage.pixel(x-5, y-1, c);
                    input[11] = myimage.pixel(x-3, y-1, c);
                    input[12] = myimage.pixel(x-1, y-1, c);
                    input[13] = myimage.pixel(x+1, y-1, c);
                    input[14] = myimage.pixel(x+3, y-1, c);
                    input[15] = myimage.pixel(x+5, y-1, c);
                    
                    input[16] = myimage.pixel(x-5, y+1, c);
                    input[17] = myimage.pixel(x-3, y+1, c);
                    input[18] = myimage.pixel(x-1, y+1, c);
                    input[19] = myimage.pixel(x+1, y+1, c);
                    input[20] = myimage.pixel(x+3, y+1, c);
                    input[21] = myimage.pixel(x+5, y+1, c);
                    
                    input[22] = myimage.pixel(x-5, y+3, c);
                    input[23] = myimage.pixel(x-3, y+3, c);
                    input[24] = myimage.pixel(x-1, y+3, c);
                    input[25] = myimage.pixel(x+1, y+3, c);
                    input[26] = myimage.pixel(x+3, y+3, c);
                    input[27] = myimage.pixel(x+5, y+3, c);
                    
                    input[28] = myimage.pixel(x-3, y+5, c);
                    input[29] = myimage.pixel(x-1, y+5, c);
                    input[30] = myimage.pixel(x+1, y+5, c);
                    input[31] = myimage.pixel(x+3, y+5, c);
                    
                    output[0] = myimage.pixel(x, y, c);
                    
                    if(useavg) avg = (input[12]+input[13]+input[18]+input[19])/4;
                    if(avg != 0.0)
                    {
                        for(int i = 0; i < sizeof(input)/sizeof(input[0]); i++)
                        {
                            input[i] /= avg;
                            if(hardavg) input[i] -= 1;
                            input[i] /= 2;
                        }
                        output[0] /= avg;
                        if(hardavg) output[0] -= 1;
                        output[0] /= 2;
                    }
                    
                    genann_train(ann, input, output, rate);
                    
                    for(int i = 0; i < 8; i++)
                        balance(&ann->weight[(32+1)*i+1]);
                }
            }
            if(y%10 == 0)
                printf("scanline %d\n", y);
        }
        rate /= 2;
    }
    
    
    // Upscale the test image
    // Our network only doubles the resolution of an image by shifting it down and to the right by half a pixel
    // So we have to do this in two passes: one to upscale diagonal between pixels, and one to upscale axial between pixels
    image outimage;
    outimage.w = myimage2.w*2-1;
    outimage.h = myimage2.h*2-1;
    outimage.n = myimage2.n;
    w = outimage.w;
    h = outimage.h;
    n = outimage.n;
    outimage.data = (unsigned char *)malloc(w*h*n);
    
    for (auto y = 0; y < h; y++)
    {
        for (auto x = 0; x < w; x++)
        {
            for (auto c = 0; c < n; c++)
            {
                int x2 = x/2;
                int y2 = y/2;
                // on pixel
                if(!(x%2) and !(y%2))
                {
                    double output = myimage2.pixel(x2, y2, c);
                    
                    outimage.data[(c)+(x*n)+(y*w*n)] = tochar(output);
                }
                // between pixels diagonally
                if(x%2 and y%2)
                {
                    double output;
                    double avg = 0;
                    
                    double input[32];
                    input[ 0] = myimage2.pixel(x2-1, y2-2, c);
                    input[ 1] = myimage2.pixel(x2-0, y2-2, c);
                    input[ 2] = myimage2.pixel(x2+1, y2-2, c);
                    input[ 3] = myimage2.pixel(x2+2, y2-2, c);
                    
                    input[ 4] = myimage2.pixel(x2-2, y2-1, c);
                    input[ 5] = myimage2.pixel(x2-1, y2-1, c);
                    input[ 6] = myimage2.pixel(x2-0, y2-1, c);
                    input[ 7] = myimage2.pixel(x2+1, y2-1, c);
                    input[ 8] = myimage2.pixel(x2+2, y2-1, c);
                    input[ 9] = myimage2.pixel(x2+3, y2-1, c);
                    
                    input[10] = myimage2.pixel(x2-2, y2-0, c);
                    input[11] = myimage2.pixel(x2-1, y2-0, c);
                    input[12] = myimage2.pixel(x2-0, y2-0, c);
                    input[13] = myimage2.pixel(x2+1, y2-0, c);
                    input[14] = myimage2.pixel(x2+2, y2-0, c);
                    input[15] = myimage2.pixel(x2+3, y2-0, c);
                    
                    input[16] = myimage2.pixel(x2-2, y2+1, c);
                    input[17] = myimage2.pixel(x2-1, y2+1, c);
                    input[18] = myimage2.pixel(x2-0, y2+1, c);
                    input[19] = myimage2.pixel(x2+1, y2+1, c);
                    input[20] = myimage2.pixel(x2+2, y2+1, c);
                    input[21] = myimage2.pixel(x2+3, y2+1, c);
                    
                    input[22] = myimage2.pixel(x2-2, y2+2, c);
                    input[23] = myimage2.pixel(x2-1, y2+2, c);
                    input[24] = myimage2.pixel(x2-0, y2+2, c);
                    input[25] = myimage2.pixel(x2+1, y2+2, c);
                    input[26] = myimage2.pixel(x2+2, y2+2, c);
                    input[27] = myimage2.pixel(x2+3, y2+2, c);
                    
                    input[28] = myimage2.pixel(x2-1, y2+3, c);
                    input[29] = myimage2.pixel(x2-0, y2+3, c);
                    input[30] = myimage2.pixel(x2+1, y2+3, c);
                    input[31] = myimage2.pixel(x2+2, y2+3, c);
                    
                    if(useavg) avg = (input[12]+input[13]+input[18]+input[19])/4;
                    if(avg != 0.0)
                    {
                        for(int i = 0; i < sizeof(input)/sizeof(input[0]); i++)
                        {
                            input[i] /= avg;
                            if(hardavg) input[i] -= 1;
                            input[i] /= 2;
                        }
                    }
                    
                    output = *genann_run(ann, input);
                    if(avg != 0.0)
                    {
                        output *= 2;
                        if(hardavg) output += 1;
                        output *= avg;
                    }
                    
                    outimage.data[(c)+(x*n)+(y*w*n)] = tochar(output);
                }
            }
        }
        printf("scanline %d\n", y);
    }
    
    for (auto y = 0; y < h; y++)
    {
        for (auto x = 0; x < w; x++)
        {
            for (auto c = 0; c < n; c++)
            {
                // between pixels axially
                if(x%2 != y%2)
                {
                    double output;
                    double avg = 0;
                    
                    double input[32];
                    input[ 0] = outimage.pixel(x+1, y-4, c);
                    input[ 1] = outimage.pixel(x+2, y-3, c);
                    input[ 2] = outimage.pixel(x+3, y-2, c);
                    input[ 3] = outimage.pixel(x+4, y-1, c);
                    
                    input[ 4] = outimage.pixel(x-1, y-4, c);
                    input[ 5] = outimage.pixel(x-0, y-3, c);
                    input[ 6] = outimage.pixel(x+1, y-2, c);
                    input[ 7] = outimage.pixel(x+2, y-1, c);
                    input[ 8] = outimage.pixel(x+3, y+0, c);
                    input[ 9] = outimage.pixel(x+4, y+1, c);
                    
                    input[10] = outimage.pixel(x-2, y-3, c);
                    input[11] = outimage.pixel(x-1, y-2, c);
                    input[12] = outimage.pixel(x-0, y-1, c);
                    input[13] = outimage.pixel(x+1, y+0, c);
                    input[14] = outimage.pixel(x+2, y+1, c);
                    input[15] = outimage.pixel(x+3, y+2, c);
                    
                    input[16] = outimage.pixel(x-3, y-2, c);
                    input[17] = outimage.pixel(x-2, y-1, c);
                    input[18] = outimage.pixel(x-1, y-0, c);
                    input[19] = outimage.pixel(x-0, y+1, c);
                    input[20] = outimage.pixel(x+1, y+2, c);
                    input[21] = outimage.pixel(x+2, y+3, c);
                    
                    input[22] = outimage.pixel(x-4, y-1, c);
                    input[23] = outimage.pixel(x-3, y-0, c);
                    input[24] = outimage.pixel(x-2, y+1, c);
                    input[25] = outimage.pixel(x-1, y+2, c);
                    input[26] = outimage.pixel(x-0, y+3, c);
                    input[27] = outimage.pixel(x+1, y+4, c);
                    
                    input[28] = outimage.pixel(x-4, y+1, c);
                    input[29] = outimage.pixel(x-3, y+2, c);
                    input[30] = outimage.pixel(x-2, y+3, c);
                    input[31] = outimage.pixel(x-1, y+4, c);
                    
                    if(useavg) avg = (input[12]+input[13]+input[18]+input[19])/4;
                    if(avg != 0.0)
                    {
                        for(int i = 0; i < sizeof(input)/sizeof(input[0]); i++)
                        {
                            input[i] /= avg;
                            if(hardavg) input[i] -= 1;
                            input[i] /= 2;
                        }
                    }
                    
                    output = *genann_run(ann, input);
                    if(avg != 0.0)
                    {
                        output *= 2;
                        if(hardavg) output += 1;
                        output *= avg;
                    }
                    
                    outimage.data[(c)+(x*n)+(y*w*n)] = tochar(output);
                }
            }
        }
        printf("scanline %d\n", y);
    }
    stbi_write_png("output.png", w, h, n, outimage.data, w*n);
    
    // Print weights for fun
    int i = 0;
    puts("\nlayer 1");
    for(int j = 0; j < (32+1)*8; j++)
        printf("%f\n", ann->weight[i++]);
    puts("\nlayer 2");
    for(int j = 0; j < (8+1)*8; j++)
        printf("%f\n", ann->weight[i++]);
    puts("\nlayer 3");
    for(int j = 0; j < (8+1)*8; j++)
        printf("%f\n", ann->weight[i++]);
    puts("\nlayer 4");
    for(int j = 0; j < (8+1)*8; j++)
        printf("%f\n", ann->weight[i++]);
    puts("\noutput");
    for(int j = 0; j < (8+1); j++)
        printf("%f\n", ann->weight[i++]);
}
