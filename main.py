'''
Author: Enrico Turato & Jason P. Beech
    with help from https://chat.openai.com/chat
    and https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    and GitHub Copilot
    and J.O.T
    and thanks to Oskar E. Ström, https://github.com/oskarestrom, for starting the project leading me to take over

Email: enrico.turato@ftf.lth.se
Date: 2024-01-24 (last code update)
Version: 1.0

Description: This script is used to split the Opto-Split videos in two halves. It takes as input the .nd2 files.
             Then it performs alignment through ORB features and generates videos. Then I analyse the videos
             to quantify the mixing of polymers-containing solutions.
'''
################################################################################################################
################################################################################################################
################################################################################################################
 

   ######      ########## ############      ######               ####     #########   ##        ##  ############
 ##      ##    ##      ##      ##         ##      ##            #    #    ##     ##   ##        ##       ##
#          #   ##      ##      ##        #          #            #        ##     ##   ##        ##       ##
#          #   #########       ##        #          #   ######     #      #########   ##        ##       ##
#          #   ##              ##        #          #                #    ##          ##        ##       ##
 ##      ##    ##              ##         ##      ##           #    #     ##          ##        ##       ##
   ######      ##              ##           ######              ####      ##          ########  ##       ##


################################################################################################################
################################################################################################################
################################################################################################################
#%%
# test cell
#%%