#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

// client side

//读写数据大小
#define MAX_LINE 4096
//监听端口
#define LINSTENPORT 8080
//服务器端口
#define SERVERPORT 8080
//缓存大小
#define BUFFSIZE 4096


void writefile(int sockfd, char *recv_file)
{
        char filename[BUFFSIZE] = {0}; //文件名
        if (recv(sockfd, filename, BUFFSIZE, 0) == -1) //connfd
        {
                perror("Can't receive filename");
                exit(1);
        }

        //创建文件
        FILE *fp = fopen(recv_file, "wb");          //filename
        if (fp == NULL)
        {
                perror("Can't open file");
                exit(1);
        }

        ssize_t n; //每次接受数据数量
        char buff[MAX_LINE] = {0}; //数据缓存
        while ((n = recv(sockfd, buff, MAX_LINE, 0)) > 0)
        {
                if (n == -1)
                {
                        perror("Receive File Error");
                        exit(1);
                }

                //将接受的数据写入文件
                if (fwrite(buff, sizeof(char), n, fp) != n)
                {
                        perror("Write File Error");
                        exit(1);
                }
                memset(buff, 0, MAX_LINE); //清空缓存
        }

        fclose(fp);
}


void sendfile(int sockfd, char *file_name)
{
        //获取文件名
        char *filename = basename(file_name); //文件名
        if (filename == NULL)
        {
                perror("Can't get filename");
                exit(1);
        }

        /*发送文件名
           为了将文件名一次发送出去，而不是暂存到TCP发送缓冲区中，避免对方收到多余的数据，不好解析正确的文件名，
           需要将要发送的数据大小设置为缓冲区大小*/
        char buff[BUFFSIZE] = {0};
        strncpy(buff, filename, strlen(filename));
        if (send(sockfd, buff, BUFFSIZE, 0) == -1)
        {
                perror("Can't send filename");
                exit(1);
        }

        //打开要发送的文件
        FILE *fp = fopen(file_name, "rb");
        if (fp == NULL)
        {
                perror("Can't open file");
                exit(1);
        }


        int n; //每次读取数据数量
        char sendline[MAX_LINE] = {0}; //暂存每次读取的数据
        while ((n = fread(sendline, sizeof(char), MAX_LINE, fp)) > 0)
        {
                if (n != MAX_LINE && ferror(fp)) //读取出错并且没有到达文件结尾
                {
                        perror("Read File Error");
                        exit(1);
                }

                //将读取的数据发送到TCP发送缓冲区
                if (send(sockfd, sendline, n, 0) == -1)
                {
                        perror("Can't send file");
                        exit(1);
                }
                memset(sendline, 0, MAX_LINE); //清空暂存字符串
        }

        //关闭文件和套接字
        fclose(fp);
}

int tcp_transfer(char *file_dir, char *op)
{
        //创建TCP套接字
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1)
        {
                perror("Can't allocate sockfd");
                exit(1);
        }

        //配置服务器套接字地址
        struct sockaddr_in clientaddr, serveraddr;
        memset(&serveraddr, 0, sizeof(serveraddr));
        serveraddr.sin_family = AF_INET;
        serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
        serveraddr.sin_port = htons(SERVERPORT);

        //绑定套接字与地址
        if (bind(sockfd, (const struct sockaddr *) &serveraddr, sizeof(serveraddr)) == -1)
        {
                perror("Bind Error");
                exit(1);
        }

        //转换为监听套接字
        if (listen(sockfd, LINSTENPORT) == -1)
        {
                perror("Listen Error");
                exit(1);
        }

        //等待连接完成
        socklen_t addrlen = sizeof(clientaddr);

        // proper server

        int connfd = accept(sockfd, (struct sockaddr *) &clientaddr, &addrlen);          //已连接套接字
        if (connfd == -1)
        {
                perror("Connect Error");
                exit(1);
        }

        close(sockfd);                          //关闭监听套接字

        if(strcmp(op, "receive") == 0) {
                writefile(connfd, file_dir);
                puts("Receive model from Server success");
        }

        if(strcmp(op, "send") == 0) {
                printf("........\n");
                sendfile(connfd, file_dir);
                puts("Send updated model back to Server success");
        }

        close(connfd);

        return 0;
}
