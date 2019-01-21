# Deep Art: Learning Artistic Style via Residual and Capsule Networks

## Project Description:

Please refer to the white paper ```DeepArt_Paper.pdf``` and the poster ```DeepArt_Poster.pdf``` for
the full details of the project.


## Prerequisites:
Make sure you install the following on your local machine
1. [git](https://gist.github.com/derhuerst/1b15ff4652a867391f03)
2. [python](https://www.python.org/downloads/)
3. [python virtualenv](https://virtualenv.pypa.io/en/stable/installation/)



## Setup:
1. Clone the giithub repository using git:
    - go on the github repo, click on clone or download and copy the repo url
    - Open an terminal and type ```git clone [copied repo]```
2. Create a [virtual env](https://virtualenv.pypa.io/en/stable/userguide/)
    -  ```virtualenv NAME_OF_YOUR_CHOICE```
    -   Activate the virtualenv : ```source NAME_OF_YOUR_CHOICE/bin/activate```
    -  install the requirements : ```pip install -r requirements.txt```
        Note: These are up for debate, it's what I ended up having in my virtualenv after the 1st coursework,
        (keras, tensorflow ext)
3. go into the DeepArt folder : ```cd DeepArt```
    - create a .gitignore file ```touch .gitignore``` : This file contains all the files an folders you want git
    to ignore, this is important as a there are a lot of things you want to keep local to you and not share with the others
    on the github repo
    - if you created the virtual enviroment on the DeepArt directory, add it to gitignore: In the .gitignore file,
    just add a line with NAME_OF_YOUR_CHOICE from before
    - Add .gitignore to .gitignore.
    - If you are using an IDE, add the .idea folder it creates to .gitignore
4. Create your branch : this will be your own working branch and all the changes you make are only visible by you until
you merge with other people.
    - ```git branch YOUR_NAME```
    - ``` git branch ``` : you should see all the created branches listed including yours. You are still on branch master though.
    - ```git checkout YOUR_NAME``` that changes the current branch to yours
    - ```git branch``` : your branch should be in green or whatever

5. Push your branch to origin (origin refers to the cloud github repo)
    - ```git push -u origin YOUR_NAME```


## Pushing changes on your branch to github:
Once you made changes locally you are happy with:
- ```git status``` : shows the status of your files- which ones you changed, created, deleted ect since last commit.
- ```git add file_you_want_to_save```  or ```git add *```:  This stages the file for commit, still not sure why we do this but its
    cool not to commit straight away. * means all changed files.
- ```git status``` : check which files have been stages for commit
- ```git commit -m "YOUR COMMIT MESSAGE"``` : Commit messages are a short phrase resuming what you changed since last commit.
- ``` git push``` : pushes your commit to github - DON'T push on the master branch, only on YOUR branch. Could ask you
for your github credentials, it is also possible to preset it and then it uses automatically - can't remember details anymore!


## Commits
Commits are basically saved versions of your code you can go back to and look, reuse, check for differences ect. They are
super useful, but can lead to nightmares if you are new to git (they still make me nervous).
I would recommend using them with caution, mostly
Few useful commands:
- ```git log``` : shows all your past commits
- ``` git checkout COMMIT_HASH_FROM_GIT_LOG``` : Takes your code back to the state it was at this last commit- WATCH
       OUT if you want to start working off this older code, can lead to headaches. I recommend using it more for comparison
       and then going back to your latest commit and working from there.
- ``` git checkout YOUR_BRANCH_NAME``` : gets you back to latest commit.
- ```git diff COMMIT_HASH1 COMMIT_HASH2``` : shows the differences in the codes between those two commits


## Pulling
You can  pull the changes made a branch to your own branch. This will update your code with whatever was in the other branch.
- ```git pull origin BRANCHNAME``` : will pull the BRANCHNAME from github .
- There can be parts where your code will clash with the BRANCHNAME's code, you will need to manually go and correct that in your files



## Workflow
The branch master is the global branch shared by the whole group.  It should not be modified directly : Everyone works on their branch.
The workflow could then be:
- Pull master (once in your branch just run ```git pull origin master```)
- Commit and push changes on your named branch - This is your playground/ experimentation place / spaggeti land.
- If something is complete and looks good enough to go on master.
- Create a new branch  OFF MASTER with branchname the feature that you are adding (E.g. Transfer_Learing_on_VGG)
- If you installed some new packages, don't forget to add them to requirements.txt
- Copy paste and clean up the code from your branch (not the most efficient but I think conceptually easier).
- Make a pull request on github with everyone as the reviewer - Check extra tutorials OR just wait until we all meet and
then merge it with master together.


## First Time using GIT?
You probably will get some errors along the way asking you to configure some git environment variables such as your name
and email, not sure what it is anymore,  but google is here! ( or the additional tutorials below)



## Additional tutorials on github and git:
- [Git, Github, Virtualenv, this seems to have lots of cool tutorials](http://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/index.html)
- [creating a pull request](https://help.github.com/articles/creating-a-pull-request/)
- [Git getting started: high level overview](https://git-scm.com/book/en/v1/Getting-Started)
- [A git guide](http://rogerdudler.github.io/git-guide/)

Ok, im just googling random stuff by now, so im gonna stop.
